'''把向量检索部署为api服务
'''
from typing import Optional, List, Union, Dict, Literal
import json
from loguru import logger
from bert4vector.core import BertSimilarity, FaissSimilarity, SameCharsSimilarity, LongestCommonSubstringSimilarity
from bert4vector.core import HownetSimilarity, SimHashSimilarity, TfidfSimilarity, BM25Similarity, CilinSimilarity
from bert4vector.snippets import cos_sim
import traceback
from torch4keras.snippets import is_package_available
if is_package_available('fastapi'):
    from fastapi import FastAPI, APIRouter, status
    from fastapi.responses import JSONResponse
    from starlette.middleware.cors import CORSMiddleware

if is_package_available('pydantic'):
    from pydantic import BaseModel, Field
else:
    BaseModel, Field = object, object


class Encode(BaseModel):
    query: Union[str, List]
    encode_kwargs:dict = {}

class Corpus(BaseModel):
    texts: Union[str, List]
    name: str = 'default'
    encode_kwargs:dict = {}

class Similarity(BaseModel):
    query1: Union[str, List]
    query2: Union[str, List]
    score_function : str="cos_sim"
    encode_kwargs:dict = {}

class Search(BaseModel):
    query: Union[str, List]
    topk: int = 10
    score_function: str = "cos_sim"
    name: str = 'default'
    encode_kwargs:dict = {}


class SimilaritySever:
    """
    Main entry point of bert search backend, start the server
    :param model_name: sentence bert model name
    :param index_dir: index dir, saved by bert_index, default bert_engine/text_index/
    :param index_name: index name, default `faiss.index`
    :param corpus_dir: corpus dir, saved by bert_embedding, default bert_engine/corpus/
    :param num_results: int, number of results to return
    :param threshold: float, threshold to return results
    :param device: pytorch device, e.g. 'cuda:0'
    :return: None, start the server

    Example:
    ```python
    >>> from bert4vector.pipelines import SimilaritySever
    >>> server = SimilaritySever('/data/pretrain_ckpt/simbert/sushen@simbert_chinese_tiny')
    >>> server.run(port=8002)
    ```
    """
    def __init__(self, 
                 model_name_or_path: str=None, 
                 mode:Literal['BertVector', 'FaissVector', 'LongestCommonSubstringSimilarity', 'HownetSimilarity',
                              'SimHashSimilarity', 'TfidfSimilarity', 'BM25Similarity', 'CilinSimilarity']='BertVector',
                 **model_config):
        if mode == 'BertVector':
            self.model = BertSimilarity(model_name_or_path, **model_config)
        elif mode == 'FaissVector':
            self.model = FaissSimilarity(model_name_or_path, **model_config)
        elif mode == 'LongestCommonSubstringSimilarity':
            self.model = LongestCommonSubstringSimilarity(**model_config)
        elif mode == 'HownetSimilarity':
            self.model = HownetSimilarity(**model_config)
        elif mode == 'SimHashSimilarity':
            self.model = SimHashSimilarity(**model_config)
        elif mode == 'TfidfSimilarity':
            self.model = TfidfSimilarity(**model_config)
        elif mode == 'BM25Similarity':
            self.model = BM25Similarity(**model_config)
        elif mode == 'CilinSimilarity':
            self.model = CilinSimilarity(**model_config)
        else:
            raise ValueError(f'Args `{mode}` not supported')
        logger.info(f'Load {mode} model success. model: {model_name_or_path}')

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
        router.add_api_route('/summary', methods=['GET'], endpoint=self.summary)  # 查询语料库目前分布
        router.add_api_route('/add_corpus', methods=['POST'], endpoint=self.add_corpus)  # 添加语料库
        router.add_api_route('/encode', methods=['POST'], endpoint=self.encode)  # 获取句向量
        router.add_api_route('/similarity', methods=['POST'], endpoint=self.similarity)  # 计算句子相似度
        router.add_api_route('/search', methods=['POST'], endpoint=self.search)  # 从语料库召回topk相似句
        self.app.include_router(router)
   
    async def index(self):
        return {"message": "index, docs url: /docs"}

    async def encode(self, req: Encode):
        '''获取query的embedding
        ## Example
        ### 入参
        ```json
        {
            "query": ["你好啊", "天气不错"]
        }
        ```
        '''
        try:
            query = req.query
            embeddings = self.model.encode(query, **req.encode_kwargs)
            msg = f"Successfully get sentence embeddings, query:{query}, res shape: {embeddings.shape}"
            logger.info(msg)
            result_dict = {'result': embeddings.tolist(), 'status': True, 'msg': msg}
            return JSONResponse(result_dict, status_code=status.HTTP_200_OK)
        except:
            msg = traceback.format_exc()
            logger.error(msg)
            return JSONResponse({'status': False, 'msg': msg}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    async def similarity(self, req:Similarity):
        '''计算两组texts之间的向量相似度
        ## Example:
        ### 入参
        ```json
        {
            "query1": ["你好啊"],
            "query2": ["你好", "天气不错"]
        }
        ```
        '''
        try:
            q1, q2 = req.query1, req.query2
            req.encode_kwargs['convert_to_numpy'] = True
            sim_score = self.model.similarity(q1, q2, score_function=req.score_function, **req.encode_kwargs)
            msg = f"Successfully get similarity score, q1:{q1}, q2:{q2}, res: {sim_score}"
            logger.info(msg)
            result_dict = {'result': sim_score.tolist(), 'status': True, 'msg': msg}
            return JSONResponse(result_dict, status_code=status.HTTP_200_OK)
        except:
            msg = traceback.format_exc()
            logger.error(msg)
            return JSONResponse({'status': False, 'msg': msg}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    async def summary(self, random_sample:bool=False, sample_count:int=2):
        return self.model.summary(random_sample, sample_count, verbose=0)

    async def add_corpus(self, req: Corpus):
        '''添加语料库
        ## Example
        ### 入参
        ```json
        {
            "texts": ["你好啊", "天气不错", "我想去北京"]
        }
        ```
        '''
        try:
            q = req.texts
            self.model.add_corpus(q, name=req.name, **req.encode_kwargs)
            msg = f"Successfully add {len(q)} texts for corpus `{req.name}` and size={len(self.model.corpus[req.name])}, all corpus size={len(self.model)}"
            logger.info(msg)
            response = {'status': True, 'msg': msg}
            return JSONResponse(response, status_code=status.HTTP_200_OK)
        except:
            msg = traceback.format_exc()
            logger.error(msg)
            return JSONResponse({'status': False, 'msg': msg}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    async def search(self, req:Search):
        ''' 在候选语料中寻找和query的向量最近似的topk个结果
        ## Example:
        ### 入参
        ```json
        {
            "query": "你好啊"
        }
        ```
        '''
        if req.name not in self.model.corpus:
            msg = f'Args `name`: `{req.name}` not in supported list: {list(self.model.corpus.keys())}'
            logger.warning(f"search error: {msg}")
            return JSONResponse({'status': False, 'msg': msg}, status_code=status.HTTP_400_BAD_REQUEST)
        try:
            query = req.query
            result = self.model.search(query, topk=req.topk, score_function=req.score_function, name=req.name, **req.encode_kwargs)
            msg = f"Successfully search from {req.name} done, query:{query}, result: {result}"
            logger.info(msg)
            result_dict = {'result': result, 'status': True, 'msg': msg}
            return JSONResponse(result_dict, status_code=status.HTTP_200_OK)
        except:
            msg = traceback.format_exc()
            logger.error(f"search error: {msg}")
            return JSONResponse({'status': False, 'msg': msg}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def run(self, *args, host: str = "0.0.0.0", port: int = 8001, **kwargs):
        ''' 开启server
        :param host: server host, default '0.0.0.0'
        :param port: server port, default 8001
        '''
        import uvicorn
        logger.info("Server starting!")
        uvicorn.run(self.app, *args, host=host, port=port, **kwargs)


class SimilarityClientRequest:
    def __init__(self, base_url: str = "http://0.0.0.0:8001", timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        import requests
        self.requests = requests

    def _post(self, endpoint: str, data: dict) -> dict:
        url = f"{self.base_url}/{endpoint}"
        try:
            response = self.requests.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except self.requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {traceback.format_exc()}")
            return {}

    def _get(self, endpoint: str, params: dict) -> dict:
        url = f"{self.base_url}/{endpoint}"
        try:
            response = self.requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except self.requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {traceback.format_exc()}")
            return {}
    
    def encode(self, query: Union[str, List], encode_kwargs:dict=None):
        data = {"query": query, 
                'encode_kwargs': encode_kwargs or dict()}
        return self._post("encode", data)

    def similarity(self, query1: Union[str, List], query2: Union[str, List], 
                   score_function : str="cos_sim", encode_kwargs:dict = None):
        data = {"query1": query1, 
                "query2": query2, 
                'score_function': score_function, 
                'encode_kwargs': encode_kwargs or dict()}
        return self._post("similarity", data)
    
    def summary(self, random_sample:bool=False, sample_count:int=2):
        params = {'random_sample': random_sample, 
                    'sample_count': sample_count}
        return self._get("summary", params)

    def search(self, query: Union[str, List], topk: int = 10,
               score_function: str = "cos_sim", name: str = 'default',
               encode_kwargs:dict = None):
        data = {"query": query, 
                'topk': topk,
                'score_function': score_function,
                'name': name,
                'encode_kwargs': encode_kwargs or dict()}
        return self._post("search", data)
    
    def add_corpus(self, texts: Union[str, List], name:str='default', encode_kwargs:dict = None):
        data = {"texts": texts, 
                'name':name, 
                'encode_kwargs': encode_kwargs or dict()}
        return self._post("add_corpus", data)


class SimilarityClientAiohttp:
    def __init__(self, base_url: str = "http://0.0.0.0:8001", timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        import aiohttp
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.aiohttp = aiohttp

    async def _post(self, endpoint: str, data: dict) -> dict:
        url = f"{self.base_url}/{endpoint}"
        try:
            headers = {"Content-type": "application/json"}
            async with self.aiohttp.ClientSession(headers=headers, timeout=self.timeout) as sess:
                async with sess.post(url, data=json.dumps(data)) as resp:
                    response = await resp.json()
            return response
        except:
            logger.error(f"Request failed for {url}: {traceback.format_exc()}")
            return {}

    async def _get(self, endpoint: str, params: dict) -> dict:
        url = f"{self.base_url}/{endpoint}"
        try:
            headers = {"Content-type": "application/json"}
            async with self.aiohttp.ClientSession(headers=headers, timeout=self.timeout) as sess:
                async with sess.get(url, params=json.dumps(params)) as resp:
                    response = await resp.json()
            return response
        except:
            logger.error(f"Request failed for {url}: {traceback.format_exc()}")
            return {}
    
    async def encode(self, query: Union[str, List], encode_kwargs:dict=None):
        data = {"query": query, 
                'encode_kwargs': encode_kwargs or dict()}
        return await self._post("encode", data)

    async def similarity(self, query1: Union[str, List], query2: Union[str, List], 
                   score_function : str="cos_sim", encode_kwargs:dict = None):
        data = {"query1": query1, 
                "query2": query2, 
                'score_function': score_function, 
                'encode_kwargs': encode_kwargs or dict()}
        return await self._post("similarity", data)
    
    async def summary(self, random_sample:bool=False, sample_count:int=2):
        params = {'random_sample': random_sample, 
                    'sample_count': sample_count}
        return await self._get("summary", params)

    async def search(self, query: Union[str, List], topk: int = 10,
               score_function: str = "cos_sim", name: str = 'default',
               encode_kwargs:dict = None):
        data = {"query": query, 
                'topk': topk,
                'score_function': score_function,
                'name': name,
                'encode_kwargs': encode_kwargs or dict()}
        return await self._post("search", data)
    
    async def add_corpus(self, texts: Union[str, List], name:str='default', encode_kwargs:dict = None):
        data = {"texts": texts, 
                'name':name, 
                'encode_kwargs': encode_kwargs or dict()}
        return await self._post("add_corpus", data)