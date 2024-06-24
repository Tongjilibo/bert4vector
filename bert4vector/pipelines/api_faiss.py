'''把向量检索部署为api服务
'''
from typing import Optional, List, Union, Dict
import os
from loguru import logger
from bert4vector.models import BertVector
from bert4vector.snippets import cos_sim
from bert4torch.snippets import is_pydantic_available
if is_pydantic_available():
    from pydantic import BaseModel, Field
else:
    BaseModel, Field = object, object


class Query(BaseModel):
    input: Union[str, List] = Field(..., max_length=512)


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
        from fastapi import FastAPI
        from fastapi import FastAPI, HTTPException, APIRouter, Depends
        from starlette.middleware.cors import CORSMiddleware

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
        router.add_api_route('/add_corpus', methods=['POST'], endpoint=self.add_corpus)  # 添加语料库
        router.add_api_route('/encode', methods=['POST'], endpoint=self.encode)  # 获取句向量
        router.add_api_route('/similarity', methods=['POST'], endpoint=self.similarity)  # 计算句子相似度
        router.add_api_route('/search', methods=['POST'], endpoint=self.search)  # 从语料库召回topk相似句
        self.app.include_router(router)
   
    async def index(self):
        return {"message": "index, docs url: /docs"}

    async def add_corpus(self, corpus: Query, batch_size: int = 32, normalize_embeddings: bool = True):
        '''添加语料库'''
        try:
            q = corpus.input
            self.model.add_corpus(q, batch_size, normalize_embeddings)
            logger.debug(f"Successfully add {len(q)} corpus")
            return {'status': True, 'msg': f"Successfully add {len(q)} corpus"}, 200
        except Exception as e:
            logger.error(e)
            return {'status': False, 'msg': e}, 400

    async def encode(self, queries: Query):
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
            q = queries.input
            embeddings = self.model.encode(q)
            result_dict = {'result': embeddings.tolist()}
            logger.debug(f"Successfully get sentence embeddings, q:{q}, res shape: {embeddings.shape}")
            return result_dict
        except Exception as e:
            logger.error(e)
            return {'status': False, 'msg': e}, 400

    async def similarity(self, query1: Query, query2: Query, score_function:str="cos_sim"):
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
            q1 = query1.input
            q2 = query2.input
            sim_score = self.model.similarity(q1, q2, score_function=score_function, convert_to_numpy=True)
            result_dict = {'result': sim_score.tolist()}
            logger.debug(f"Successfully get similarity score, q1:{q1}, q2:{q2}, res: {sim_score}")
            return result_dict
        except Exception as e:
            logger.error(e)
            return {'status': False, 'msg': e}, 400

    async def search(self, queries: Query, topk:int=10, score_function:str="cos_sim"):
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
            q = queries.input
            result = self.model.search(q, topk=topk, score_function=score_function)
            result_dict = {'result': result}
            logger.debug(f"Successfully search done, q:{q}, res size: {len(result)}")
            return result_dict
        except Exception as e:
            logger.error(f"search error: {e}")
            return {'status': False, 'msg': e}, 400

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