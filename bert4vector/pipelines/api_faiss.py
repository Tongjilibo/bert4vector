'''把向量检索部署为api服务
'''
from typing import Optional, List, Union, Dict
import os
from loguru import logger
from bert4vector.models import BertVector
from bert4vector.snippets import cos_sim
from bert4torch.snippets import is_pydantic_available, is_fastapi_available
if is_fastapi_available():
    from fastapi import FastAPI, APIRouter, status
    from fastapi.responses import JSONResponse
    from starlette.middleware.cors import CORSMiddleware

if is_pydantic_available():
    from pydantic import BaseModel, Field
else:
    BaseModel, Field = object, object


class Query(BaseModel):
    query: Union[str, List]

class Corpus(BaseModel):
    texts: Union[str, List]
    batch_size: int = 32
    normalize_embeddings: bool = True
    name: str = 'default'

class Similarity(BaseModel):
    query1: Union[str, List]
    query2: Union[str, List]
    score_function : str="cos_sim"

class Search(BaseException):
    query: Union[str, List]
    topk: int = 10
    score_function: str = "cos_sim"
    name: str = 'default'

class EmbeddingSever:
    """
    Main entry point of bert search backend, start the server
    :param model_name: sentence bert model name
    :param index_dir: index dir, saved by bert_index, default bert_engine/text_index/
    :param index_name: index name, default `faiss.index`
    :param corpus_dir: corpus dir, saved by bert_embedding, default bert_engine/corpus/
    :param num_results: int, number of results to return
    :param threshold: float, threshold to return results
    :param device: pytorch device, e.g. 'cuda:0'
    :param debug: whether to print debug info, default False
    :return: None, start the server

    Example:
    ```python
    >>> from bert4vector.pipelines import EmbeddingSever
    >>> server = EmbeddingSever('/data/pretrain_ckpt/simbert/sushen@simbert_chinese_tiny')
    >>> server.run(port=8002)
    ```
    """
    def __init__(self, model_name_or_path: str, debug: bool = False, **model_config):
        logger.info("starting boot of bert server")
        self.model = BertVector(model_name_or_path, **model_config)
        logger.info(f'Load model success. model: {model_name_or_path}')

        # define the app
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"])
        
        router = APIRouter()
        router.add_api_route('/', methods=['GET'], endpoint=self.index)
        router.add_api_route('/summary', methods=['POST'], endpoint=self.summary)  # 查询语料库目前分布
        router.add_api_route('/add_corpus', methods=['POST'], endpoint=self.add_corpus)  # 添加语料库
        router.add_api_route('/encode', methods=['POST'], endpoint=self.encode)  # 获取句向量
        router.add_api_route('/similarity', methods=['POST'], endpoint=self.similarity)  # 计算句子相似度
        router.add_api_route('/search', methods=['POST'], endpoint=self.search)  # 从语料库召回topk相似句
        self.app.include_router(router)
   
    async def index(self):
        return {"message": "index, docs url: /docs"}

    async def summary(self, random_sample:bool=False, sample_count:int=2):
        return self.model.summary(random_sample, sample_count, verbose=0)

    async def add_corpus(self, req: Corpus):
        '''添加语料库'''
        try:
            q = req.texts
            self.model.add_corpus(q, batch_size=req.batch_size, name=req.name, 
                                  normalize_embeddings=req.normalize_embeddings)
            msg = f"Successfully add {len(q)} texts for corpus `{req.name}`, total_size={len(self.model.corpus[req.name])}"
            logger.debug(msg)
            response = {'status': True, 'msg': msg}
            return JSONResponse(response, status_code=status.HTTP_200_OK)
        except Exception as e:
            logger.error(e)
            return JSONResponse({'status': False, 'msg': e}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    async def encode(self, req: Query):
        '''获取query的embedding
        ## Example
        ### 入参
        ```json
        {
            "input": ["你好啊"]
        }
        ```

        ### 出参
        ```json
        {
            "result": [
                [
                    -0.0827251672744751,
                    0.05515166372060776,
                    -0.034269820898771286,
                    -0.13995569944381714,
                    0.06154867261648178,
                    0.040889400988817215,
                    -0.06582749634981155
                ]
                    ]
        }
        ```
        '''
        try:
            q = req.query
            embeddings = self.model.encode(q)
            msg = f"Successfully get sentence embeddings, q:{q}, res shape: {embeddings.shape}"
            logger.debug(msg)
            result_dict = {'result': embeddings.tolist(), 'status': True, 'msg': msg}
            return JSONResponse(result_dict, status_code=status.HTTP_200_OK)
        except Exception as e:
            logger.error(e)
            return JSONResponse({'status': False, 'msg': e}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    async def similarity(self, req:Similarity):
        '''计算两组texts之间的向量相似度
        ## Example:
        ### 入参
        ```json
        {
            "query1": {"input": ["你好啊"]},
            "query2": {"input": ["你好", "天气不错"]}
        }
        ```

        ### 出参
        ```json
        {
            "result": [
                [
                    0.9075419902801514,
                    0.1958463490009308
                ]
            ]
        }
        ```
        '''
        try:
            q1, q2 = req.query1, req.query2
            sim_score = self.model.similarity(q1, q2, score_function=req.score_function, convert_to_numpy=True)
            msg = f"Successfully get similarity score, q1:{q1}, q2:{q2}, res: {sim_score}"
            logger.debug(msg)
            result_dict = {'result': sim_score.tolist(), 'status': True, 'msg': msg}
            return JSONResponse(result_dict, status_code=status.HTTP_200_OK)
        except Exception as e:
            logger.error(e)
            return JSONResponse({'status': False, 'msg': e}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    async def search(self, req:Search):
        ''' 在候选语料中寻找和query的向量最近似的topk个结果
        ## Example:
        ### 入参
        ```json
        {
            "input": "你好啊"
        }
        ```

        ### 出参

        '''
        try:
            q = req.query
            result = self.model.search(q, topk=req.topk, score_function=req.score_function, name=req.name)
            msg = f"Successfully search from {req.name} done, q:{q}, res size: {len(result)}"
            logger.debug(msg)
            result_dict = {'result': result, 'status': True, 'msg': msg}
            return JSONResponse(result_dict, status_code=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"search error: {e}")
            return JSONResponse({'status': False, 'msg': e}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def run(self, *args, host: str = "0.0.0.0", port: int = 8001, **kwargs):
        ''' 开启server
        :param host: server host, default '0.0.0.0'
        :param port: server port, default 8001
        '''
        import uvicorn
        logger.info("Server starting!")
        uvicorn.run(self.app, *args, host=host, port=port, **kwargs)


class EmbeddingClient:
    def __init__(self, base_url: str = "http://0.0.0.0:8001", timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        import requests
        self.requests = requests

    def _post(self, endpoint: str, data: dict) -> dict:
        try:
            response = self.requests.post(f"{self.base_url}/{endpoint}", json=data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except self.requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {}

    def emb(self, input_text: str) -> List[float]:
        try:
            data = {"input": input_text}
            response = self._post("emb", data)
            return response.get("emb", [])
        except Exception as e:
            logger.error(f"get_emb error: {e}")
            return []

    def similarity(self, input_text1: str, input_text2: str) -> float:
        try:
            data1 = {"input": input_text1}
            data2 = {"input": input_text2}
            response = self._post("similarity", {"item1": data1, "item2": data2})
            return response.get("result", 0.0)
        except Exception as e:
            logger.error(f"get_similarity error: {e}")
            return 0.0

    def search(self, input_text: str):
        try:
            data = {"input": input_text}
            response = self._post("search", data)
            return response.get("result", [])
        except Exception as e:
            logger.error(f"search error: {e}")
            return []