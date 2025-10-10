from typing import Optional, List, Union, Dict, Callable, Any
import json
from loguru import logger
from bert4vector.core.base import SimilarityBase, VectorSimilarity, SEARCH_RES_TYPE
import numpy as np
from torch4keras.snippets import is_package_available
from bert4vector.snippets import cos_sim, dot_score, semantic_search
import requests


if is_package_available('torch'):
    import torch  

# 默认 pre_process: OpenAI-compatible 格式
def default_pre_proc(sentences, model_name):
    inputs = [sentences] if isinstance(sentences, str) else sentences
    return {
        "model": model_name,
        "input": inputs,
        "encoding_format": "float"
    }


# 默认 post_process: 提取 data[i].embedding
def default_post_proc(response_json):
    if "data" not in response_json:
        raise ValueError(f"Invalid response format: missing 'data' field. Got: {list(response_json.keys())}")
    return [item["embedding"] for item in sorted(response_json["data"], key=lambda x: x.get("index", 0))]


# 默认 headers
def default_headers(api_key):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


class OpenaiSimilarityRequest(VectorSimilarity):
    """ 在内存中存储和检索向量，访问远程的模型
    :param model_url: 模型权重地址
    :param config_path: 权重的config地址
    :param corpus: Corpus of documents to use for similarity queries.
    :param device: Device (like 'cuda' / 'cpu') to use for the computation.

    Example:
    ```python
    >>> from bert4vector.core import OpenaiSimilarityRequest
    >>> model = OpenaiSimilarityRequest('http://10.16.38.1:9901/v1/embeddings', 'bge-m3')
    >>> model.add_corpus(['你好', '我选你'])
    >>> model.add_corpus(['天气不错', '人很好看'])
    >>> print(model.search('你好', topk=2))
    >>> print(model.search(['你好', '天气晴']))

    >>> # {'你好': [{'corpus_id': 0, 'score': 0.9999, 'text': '你好'},
    ... #           {'corpus_id': 3, 'score': 0.5694, 'text': '人很好看'}]} 
    ```
    """
    def __init__(
        self, 
        model_url:str, 
        model_name:str, 
        api_key:str=None, 
        corpus: List[str] = None, 
        timeout: int = 30, 
        # 自定义函数：输入 sentences, model_name → 返回 dict (payload)
        pre_process_func: Callable[[Union[str, List[str]], str], Dict[Any, Any]] = None,
        # 自定义函数：输入 response.json() → 返回 List[List[float]] (embeddings)
        post_process_func: Callable[[Dict[Any, Any]], List[List[float]]] = None,
        # 可选：自定义 headers 构造函数
        headers_func: Callable[[Optional[str]], Dict[str, str]] = None,
        **model_config
    ):
        self.model_url = model_url
        self.model_name = model_name
        self.timeout = timeout
        self.pre_process_func = pre_process_func or default_pre_proc
        self.post_process_func = post_process_func or default_post_proc
        self.headers = (headers_func or default_headers)(api_key)

        super().__init__(corpus=corpus)
        self.emb_path = "corpus_emb.jsonl"
    def encode(
        self,
        sentences: Union[str, List[str]],
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        **kwargs
    ):
        """ 把句子转换成向量
        """
        # TODO:
        payload = self.pre_process_func(sentences, self.model_name)

        try:
            response = requests.post(self.model_url, headers=self.headers, data=json.dumps(payload), 
                                     timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            raise
        
        all_embeddings = self.post_process_func(data)

        if convert_to_numpy:
            all_embeddings = np.asarray(all_embeddings, dtype=np.float32)
        elif convert_to_tensor:
            all_embeddings = torch.tensor(all_embeddings, dtype=torch.float32)

        return all_embeddings
    

class OpenaiSimilarityAiohttp(SimilarityBase):
    """ 在内存中存储和检索向量，访问远程的模型
    :param model_url: 模型权重地址
    :param config_path: 权重的config地址
    :param corpus: Corpus of documents to use for similarity queries.
    :param device: Device (like 'cuda' / 'cpu') to use for the computation.

    Example:
    ```python
    >>> from bert4vector.core import OpenaiSimilarityAiohttp
    >>> model = OpenaiSimilarityAiohttp('http://10.16.38.1:9901/v1/embeddings', 'bge-m3')
    >>> await model.add_corpus(['你好', '我选你'])
    >>> await model.add_corpus(['天气不错', '人很好看'])
    >>> print(await model.search('你好', topk=2))
    >>> print(await model.search(['你好', '天气晴']))

    >>> # {'你好': [{'corpus_id': 0, 'score': 0.9999, 'text': '你好'},
    ... #           {'corpus_id': 3, 'score': 0.5694, 'text': '人很好看'}]} 
    ```
    """
    def __init__(
        self, 
        model_url:str, 
        model_name:str, 
        api_key:str=None, 
        corpus: List[str] = None, 
        timeout: int = 30, 
        # 自定义函数：输入 sentences, model_name → 返回 dict (payload)
        pre_process_func: Callable[[Union[str, List[str]], str], Dict[Any, Any]] = None,
        # 自定义函数：输入 response.json() → 返回 List[List[float]] (embeddings)
        post_process_func: Callable[[Dict[Any, Any]], List[List[float]]] = None,
        # 可选：自定义 headers 构造函数
        headers_func: Callable[[Optional[str]], Dict[str, str]] = None,
        **model_config
    ):
        self.model_url =  model_url
        self.model_name = model_name
        self.pre_process_func = pre_process_func or default_pre_proc
        self.post_process_func = post_process_func or default_post_proc
        self.headers = (headers_func or default_headers)(api_key)

        import aiohttp
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.aiohttp = aiohttp

        super().__init__(corpus=corpus)
        self.emb_path = "corpus_emb.jsonl"
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}

    async def encode(
        self,
        sentences: Union[str, List[str]],
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        **kwargs
    ):
        """ 把句子转换成向量
        """
        payload = self.pre_process_func(sentences, self.model_name)

        try:
            async with self.aiohttp.ClientSession(headers=self.headers, timeout=self.timeout) as sess:
                async with sess.post(self.model_url, data=json.dumps(payload)) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            raise

        all_embeddings = self.post_process_func(data)

        if convert_to_numpy:
            all_embeddings = np.asarray(all_embeddings, dtype=np.float32)
        elif convert_to_tensor:
            all_embeddings = torch.tensor(all_embeddings, dtype=torch.float32)

        return all_embeddings
    
    async def add_corpus(self, corpus: List[str], name:str='default', corpus_property:List[dict]=None, **kwargs):
        """ 使用文档chunk来转为向量
        :param corpus: 语料的list
        :param name: sub_corpus名
        :param corpus_property: 语料属性，由于corpus是去重时候会结果不同，为了防止和外部属性对应不上，这里可以返回
        """
        # 添加语料并放到语料库
        new_corpus = self._add_corpus(corpus=corpus, name=name, corpus_property=corpus_property)
        
        # 转向量并放到向量库
        await self._add_embedding(corpus=[item['text'] for item in new_corpus.values()], name=name, **kwargs)

        # log
        msg = f"Add {len(new_corpus)} docs for `{name}`, total: {len(self.corpus[name])}"
        if len(self.corpus_embeddings[name]) > 0:
            msg += f", emb dim: {len(self.corpus_embeddings[name][0])}"
        logger.info(msg)

    async def _add_embedding(self, corpus:List[str], name:str='default', **encode_kwargs):
        """ 使用文档chunk来转为向量

        >>> encode_kwargs参数
        :param batch_size: batch size for computing embeddings
        :param normalize_embeddings: normalize embeddings before computing similarity
        """
        # 转向量并放到向量库
        corpus_embeddings = await self.encode(corpus, convert_to_numpy=False, show_progress_bar=True, **encode_kwargs)
        self.corpus_embeddings[name] = self.corpus_embeddings.get(name, []) + corpus_embeddings

    async def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]], score_function:str="cos_sim", **encode_kwargs):
        """ 计算两组texts之间的向量相似度
        :param a: list of str or str
        :param b: list of str or str
        :param score_function: function to compute similarity, default cos_sim
        :param kwargs: additional arguments for the similarity function
        :return: similarity score, torch.Tensor, Matrix with res[i][j] = cos_sim(a[i], b[j])
        """
        if score_function not in self.score_functions:
            raise ValueError(f"score function: {score_function} must be either (cos_sim) for cosine similarity"
                             " or (dot) for dot product")
        score_function = self.score_functions[score_function]
        text_emb1 = await self.encode(a, **encode_kwargs)
        text_emb2 = await self.encode(b, **encode_kwargs)

        return score_function(text_emb1, text_emb2)

    async def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """计算两组texts之间的cos距离"""
        return 1 - await self.similarity(a, b)

    async def search(self, queries: Union[str, List[str]], topk:int=10, score_function:str="cos_sim", name:str='default', return_dict:bool=True, **encode_kwargs) -> SEARCH_RES_TYPE:
        """ 在候选语料中寻找和query的向量最近似的topk个结果
        :param queries:str or list of str
        :param topk: int
        :param score_function: function to compute similarity, default cos_sim
        :param name: sub_corpus名
        :param encode_kwargs: additional arguments for the similarity function

        :return: Dict[str, Dict[str, float]], {query_id: {corpus_id: similarity_score}, ...}
        """

        queries, queries_embeddings = await self._get_query_emb(queries, **encode_kwargs)
        if score_function not in self.score_functions:
            raise ValueError(f"score function: {score_function} must be either (cos_sim) for cosine similarity"
                             " or (dot) for dot product")
        score_function = self.score_functions[score_function]
        corpus_embeddings = np.array(self.corpus_embeddings[name], dtype=np.float32)
        all_hits = semantic_search(queries_embeddings, corpus_embeddings, top_k=topk, score_function=score_function)
        
        result = {}
        for idx, hits in enumerate(all_hits):
            items = []
            for hit in hits[0:topk]:
                corpus_id = hit['corpus_id']
                items.append({**self.corpus[name][corpus_id], **hit})
            result[queries[idx]] = items
        return self._get_search_result(result, return_dict)

    async def _get_query_emb(self, queries:Union[str, List[str]], **encode_kwargs):
        '''获取query的句向量'''
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        queries_embeddings = await self.encode(queries, **encode_kwargs)
        return queries, queries_embeddings
