from typing import Dict, List, Union
from bert4vector.bert import BertVector
import importlib
import math
import numpy as np
from loguru import logger


if importlib.util.find_spec('faiss') is not None:
    import faiss

class FaissVector(BertVector):
    ''' 用faiss来存储和检索向量
    Example:
    ```python
    >>> from bert4vector import FaissVector
    >>> model = FaissVector('E:/pretrain_ckpt/simbert/sushen@simbert_chinese_tiny')
    >>> model.add_corpus(['你好', '我选你'], gpu_index=True)
    >>> model.add_corpus(['天气不错', '人很好看'], gpu_index=True)
    >>> print(model.search('你好'))
    >>> print(model.search(['你好', '天气晴']))
    ```
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert hasattr(faiss, "IndexFlatIP")

    def reset(self):
        '''重置向量库'''
        super().reset()
        self.index = None
    
    def add_corpus(self, *args, ann_search:bool=False, gpu_index:bool=False,
                   gpu_memory:int=16, n_search:int=64, **kwargs):
        ''' 使用文档chunk来转为向量
        :param corpus: 语料的list
        :param batch_size: batch size for computing embeddings
        :param normalize_embeddings: normalize embeddings before computing similarity
        :param ann_search: 使用ANN搜索算法
        :param gpu_index: 计算使用的gpu卡id
        :param gpu_memory: gpu的显存设置
        :param n_search: IndexIVFFlat 的 nprobe 属性默认为1, 在nprobe个最近邻的簇向量空间中进行 k 近邻搜索
        '''
        super().add_corpus(*args, **kwargs)
        d = len(self.corpus_embeddings[0])
        nlist = int(math.sqrt(len(self.corpus_embeddings)))
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
        
        corpus_embeddings = np.array(self.corpus_embeddings, dtype=np.float32)
        if ann_search:
            index.train(corpus_embeddings)
        index.add(corpus_embeddings)
        index.nprobe = min(nlist, n_search)
        self.index = index
    
    def save_embeddings(self, emb_path:str="faiss_emb.index"):
        '''把corpus_embeddings保存到本地'''
        faiss.write_index(self.index, emb_path)
    
    def load_embeddings(self, emb_path:str="faiss_emb.index"):
        '''从本地加载corpus_embeddings'''
        self.index = faiss.read_index(emb_path)
    
    def most_similar(self, queries: Union[List[str], Dict[str, str]], topk:int=10, score_function:str="cos_sim", **kwargs):
        ''' 在候选语料中寻找和query的向量最近似的topk个结果
        Example:
        ```python
        >>> from bert4vector import FaissVector
        >>> model = FaissVector('/data/pretrain_ckpt/simbert/sushen@simbert_chinese_tiny')
        >>> model.add_corpus(['你好', '我选你'], gpu_index=True)
        >>> model.add_corpus(['天气不错', '人很好看'], gpu_index=True)
        >>> print(model.most_similar('你好'))
        >>> print(model.most_similar(['你好', '天气晴']))
        '''
        queries, queries_embeddings, queries_ids_map = super().get_query_emb(queries, **kwargs)
        distance, idx = self.index.search(np.array(queries_embeddings.cpu(), dtype=np.float32), topk)
        
        results = {}
        for id_, (i, s) in enumerate(zip(idx, distance)):
            items = []
            for j, k in zip(i, s):
                if j < 0:
                    continue
                items.append({'text': self.corpus[j], 'corpus_id': j, 'score': k})
            results[queries[queries_ids_map[id_]]] = items
        return results
