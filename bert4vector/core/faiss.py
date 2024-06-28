from typing import Dict, List, Union
from .bert import BertSimilarity
import importlib
import math
import numpy as np
from loguru import logger
import os


if importlib.util.find_spec('faiss') is not None:
    import faiss

class FaissSimilarity(BertSimilarity):
    ''' 用faiss来存储和检索向量
    Example:
    ```python
    >>> from bert4vector import FaissSimilarity
    >>> model = FaissSimilarity('E:/pretrain_ckpt/simbert/sushen@simbert_chinese_tiny')
    >>> model.add_corpus(['你好', '我选你'], gpu_index=True)
    >>> model.add_corpus(['天气不错', '人很好看'], gpu_index=True)
    >>> print(model.search('你好'))
    >>> print(model.search(['你好', '天气晴']))
    ```
    '''
    def __init__(self, *args, **kwargs):
        self.indexes = dict()
        super().__init__(*args, **kwargs)
        assert hasattr(faiss, "IndexFlatIP")
        self.emb_path = "faiss_emb.index"

    def reset(self):
        '''重置向量库'''
        super().reset()
        self.indexes = dict()
    
    def _add_embedding(self, new_corpus: Dict[int, str], name:str='default', 
                   ann_search:bool=False, gpu_index:bool=False,
                   gpu_memory:int=16, n_search:int=64, **kwargs):
        ''' 使用文档chunk来转为向量
        :param corpus: 语料的list
        :param batch_size: batch size for computing embeddings
        :param normalize_embeddings: normalize embeddings before computing similarity
        :param name: sub_corpus名
        :param ann_search: 使用ANN搜索算法
        :param gpu_index: 计算使用的gpu卡id
        :param gpu_memory: gpu的显存设置
        :param n_search: IndexIVFFlat 的 nprobe 属性默认为1, 在nprobe个最近邻的簇向量空间中进行 k 近邻搜索
        '''
        super()._add_embedding(new_corpus, name=name, **kwargs)
        d = len(self.corpus_embeddings[name][0])
        nlist = int(math.sqrt(len(self.corpus_embeddings[name])))
        quantizer = faiss.IndexFlatIP(d)
        if ann_search:
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            index = quantizer 
        
        if gpu_index:
            if hasattr(faiss, "StandardGpuResources"):
                logger.info("Use GPU-version faiss")
                res = faiss.StandardGpuResources()
                res.setTempMemory(gpu_memory * 1024 * 1024 * 1024)
                index = faiss.index_cpu_to_gpu(res, 0, index)
            else:
                logger.info("Use CPU-version faiss")
        
        corpus_embeddings = np.array(self.corpus_embeddings[name], dtype=np.float32)
        if ann_search:
            index.train(corpus_embeddings)
        index.add(corpus_embeddings)
        index.nprobe = min(nlist, n_search)
        self.indexes[name] = index
    
    def _save_embeddings(self, emb_path:str=None):
        '''把corpus_embeddings保存到本地'''
        emb_path = self.emb_path if emb_path is None else emb_path
        for name, index in self.indexes.items():
            faiss.write_index(index, emb_path + '.' + name)

    def _load_embeddings(self, emb_path:str=None):
        '''从本地加载corpus_embeddings'''
        emb_path = self.emb_path if emb_path is None else emb_path
        path = os.path.dirname(emb_path)
        file = os.path.basename(emb_path)
        for file_name in os.listdir(path):
            if file + '.' in file_name:
                name = file_name.replace(file + '.', '')
                self.indexes[name] = faiss.read_index(os.path.join(path, file_name))

    def search(self, queries: Union[str, List[str]], topk:int=10, name:str='default', **kwargs) -> dict:
        ''' 在候选语料中寻找和query的向量最近似的topk个结果
        :param queries: query语句/语句列表/语句字典
        :param topk: 对每条query需要召回topk条
        :param name: sub_corpus名

        Example:
        ```python
        >>> from bert4vector import FaissSimilarity
        >>> model = FaissSimilarity('/data/pretrain_ckpt/simbert/sushen@simbert_chinese_tiny')
        >>> model.add_corpus(['你好', '我选你'], gpu_index=True)
        >>> model.add_corpus(['天气不错', '人很好看'], gpu_index=True)
        >>> print(model.search('你好', topk=2))
        >>> print(model.search(['你好', '天气晴']))

        >>> # {'你好': [{'corpus_id': 0, 'score': 1.406428, 'text': '你好'},
        ... #           {'corpus_id': 3, 'score': 0.800828, 'text': '人很好看'}]} 
        '''
        queries, queries_embeddings = super()._get_query_emb(queries, **kwargs)
        distance, idx = self.indexes[name].search(queries_embeddings, topk)
        
        results = {}
        for idx, (i, s) in enumerate(zip(idx, distance)):
            items = []
            for j, k in zip(i, s):
                if j < 0:
                    continue
                items.append({'text': self.corpus[name][j], 'corpus_id': j, 'score': k})
            results[queries[idx]] = items
        return results
