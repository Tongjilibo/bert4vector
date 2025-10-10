from typing import Optional, List, Union, Dict, Callable, Any
import json
from loguru import logger
from bert4vector.core.base import VectorSimilarity, AsyncVectorSimilarity, SEARCH_RES_TYPE
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
    

class OpenaiSimilarityAiohttp(AsyncVectorSimilarity):
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
    