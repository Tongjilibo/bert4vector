from loguru import logger
from typing import List, Union, Dict, Literal
import numpy as np
import json
from .base import SimilarityBase
from bert4vector.snippets import cos_sim, dot_score, semantic_search
from pathlib import Path


class BertSimilarity(SimilarityBase):
    """ 在内存中存储和检索向量
    :param checkpoint_path: 模型权重地址
    :param config_path: 权重的config地址
    :param corpus: Corpus of documents to use for similarity queries.
    :param device: Device (like 'cuda' / 'cpu') to use for the computation.
    """
    def __init__(self, model_name_or_path:str, model_type:Literal['bert4torch', 'sentence_transformers']='bert4torch', 
                 corpus: List[str] = None, matching_type:str='BertSimilarity', **model_config):
        self.model_type = model_type
        self.model = self.build_model(model_name_or_path, **model_config)
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        super().__init__(corpus=corpus, matching_type=matching_type)

    def build_model(self, model_name_or_path, **model_config):
        '''初始化模型'''
        if self.model_type == 'bert4torch':
            from bert4torch.pipelines import Text2Vec
            return Text2Vec(model_name_or_path, **model_config)
        elif self.model_type == 'sentence_transformers':
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(model_name_or_path, **model_config)
        else:
            raise ValueError(f'Args `model_type` {self.model_type} not supported')
    
    def to(self, device):
        self.model.to(device)
        self.device = device

    def encode(
            self,
            sentences: Union[str, List[str]],
            batch_size: int = 8,
            show_progress_bar: bool = False,
            pool_strategy=None,
            custom_layer=None,
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            normalize_embeddings: bool = False,
            max_seq_length: int = None
    ):
        """ 把句子转换成向量
        Returns the embeddings for a batch of sentences.
        
        Example:
        ```python
        >>> from bert4vector import BertVector
        >>> model = BertVector('/data/pretrain_ckpt/simbert/sushen@simbert_chinese_tiny')   
        >>> sentences = ['喜欢打篮球的男生喜欢什么样的女生', '西安下雪了？是不是很冷啊?', 
        ...              '第一次去见女朋友父母该如何表现？', '小蝌蚪找妈妈怎么样', 
        ...              '给我推荐一款红色的车', '我喜欢北京']
        >>> vecs = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=False)
        >>> print(vecs.shape)
        >>> print(vecs)
        """
        return self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            pool_strategy=pool_strategy,
            custom_layer=custom_layer,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
            normalize_embeddings=normalize_embeddings,
            max_seq_length=max_seq_length
        )
    
    def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]], score_function:str="cos_sim", **encode_kwargs):
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
        text_emb1 = self.encode(a, **encode_kwargs)
        text_emb2 = self.encode(b, **encode_kwargs)

        return score_function(text_emb1, text_emb2)

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """计算两组texts之间的cos距离"""
        return 1 - self.similarity(a, b)

    def _add_embedding(self, new_corpus: Dict[int, str], name:str='default', **encode_kwargs):
        """ 使用文档chunk来转为向量

        >>> encode_kwargs参数
        :param batch_size: batch size for computing embeddings
        :param normalize_embeddings: normalize embeddings before computing similarity
        """
        # 转向量并放到向量库
        corpus_embeddings = self.encode(
            list(new_corpus.values()),
            show_progress_bar=True,
            **encode_kwargs
        ).tolist()
        self.corpus_embeddings[name] = self.corpus_embeddings[name] + corpus_embeddings
    
    def search(self, queries: Union[str, List[str]], topk:int=10, score_function:str="cos_sim", name:str='default', **encode_kwargs) -> Dict[str, List]:
        """ 在候选语料中寻找和query的向量最近似的topk个结果
        :param queries:str or list of str
        :param topk: int
        :param score_function: function to compute similarity, default cos_sim
        :param name: sub_corpus名
        :param encode_kwargs: additional arguments for the similarity function

        :return: Dict[str, Dict[str, float]], {query_id: {corpus_id: similarity_score}, ...}

        Example:
        ```python
        >>> from bert4vector import BertVector
        >>> model = BertVector('/data/pretrain_ckpt/simbert/sushen@simbert_chinese_tiny')
        >>> model.add_corpus(['你好', '我选你'], gpu_index=True)
        >>> model.add_corpus(['天气不错', '人很好看'], gpu_index=True)
        >>> print(model.search('你好', topk=2))
        >>> print(model.search(['你好', '天气晴']))

        >>> # {'你好': [{'corpus_id': 0, 'score': 0.9999, 'text': '你好'},
        ... #           {'corpus_id': 3, 'score': 0.5694, 'text': '人很好看'}]} 

        """

        queries, queries_embeddings = self._get_query_emb(queries, **encode_kwargs)
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
                items.append({**{'text': self.corpus[name][corpus_id]}, **hit})
            result[queries[idx]] = items
        return result
    
    def _get_query_emb(self, queries:Union[str, List[str]], **encode_kwargs):
        '''获取query的句向量'''
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        queries_embeddings = self.encode(queries, convert_to_tensor=True, **encode_kwargs)
        return queries, queries_embeddings

    def _save_embeddings(self, emb_path:Path=None):
        """ 把语料向量保存到json文件中
        :param emb_path: json file path
        :return:
        """
        emb_path = "corpus_emb.json" if emb_path is None else emb_path
        corpus_emb = dict()
        for name, sub_corpus in self.corpus.items():
            corpus_emb[name] = {id: {"doc": sub_corpus[id], "doc_emb": emb} for id, emb in
                                zip(sub_corpus.keys(), self.corpus_embeddings[name])}
        with open(emb_path, "w", encoding="utf-8") as f:
            json.dump(corpus_emb, f, ensure_ascii=False)
        logger.info(f'Successfully save embeddings: {emb_path}')

    def _load_embeddings(self, emb_path:Path=None):
        """ 从json文件中加载语料向量
        :param emb_path: json file path
        :return: list of corpus embeddings, dict of corpus ids map, dict of corpus
        """
        emb_path = "corpus_emb.json" if emb_path is None else emb_path
        with open(emb_path, "r", encoding="utf-8") as f:
            corpus_emb = json.load(f)
        corpus_embeddings = dict()
        for name, sub_corpus_emb in corpus_emb.items():
            sub_corpus_embeddings = []
            for id, corpus_dict in sub_corpus_emb.items():
                self.corpus[int(id)] = corpus_dict["doc"]
                sub_corpus_embeddings.append(corpus_dict["doc_emb"])
            corpus_embeddings[name] = sub_corpus_embeddings
        self.corpus_embeddings = corpus_embeddings
        logger.info(f'Successfully load embeddings: {emb_path}')
