from typing import Dict, List, Union
from bert4vector.bert import BertVector
import importlib
import math
import numpy as np
from loguru import logger


if importlib.util.find_spec('faiss') is not None:
    import faiss

class FaissVector(BertVector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert hasattr(faiss, "IndexFlatIP")

    def reset(self):
        '''重置向量库'''
        super().reset()
        self.index = None
    
    def add_corpus(self, *args, ann_search:bool=False, gpu_index:bool=False,
                   gpu_memory:int=16, n_search:int=64, **kwargs):
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
        faiss.write_index(self.index, emb_path)
    
    def load_embeddings(self, emb_path:str="faiss_emb.index"):
        self.index = faiss.read_index(emb_path)
    
    def most_similar(self, queries: Union[List[str], Dict[str, str]], topk:int=10, score_function:str="cos_sim", **kwargs):
        queries, queries_embeddings, queries_ids_map = super().get_query_emb(queries, **kwargs)
        distance, idx = self.index.search(np.array(queries_embeddings, dtype=np.float32), topk)
        
        results = {}
        for id_, (i, s) in enumerate(zip(idx, distance)):
            items = []
            for j, k in zip(i, s):
                if j < 0:
                    continue
                items.append({'text': self.corpus[j], 'corpus_id': j, 'score': k})
            results[queries[queries_ids_map[id_]]] = items
        return results


class SentTransformersFaissVector(FaissVector):
    def build_model(self, model_path, **model_config):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_path)