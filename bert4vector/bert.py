from loguru import logger
from typing import List, Union, Dict
import numpy as np
import json
from bert4torch.pipelines import Text2Vec
from bert4vector.base import Base
from bert4vector.utils import cos_sim, dot_score, semantic_search


class BertVector(Base):
    def __init__(self, model_path, corpus: Union[List[str], Dict[str, str]] = None, **model_config):
        """
        Initialize the similarity object.
        :param checkpoint_path: 模型权重地址
        :param config_path: 权重的config地址
        :param corpus: Corpus of documents to use for similarity queries.
        :param device: Device (like 'cuda' / 'cpu') to use for the computation.
        """
        self.model = Text2Vec(model_path, **model_config)
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.corpus = {}
        self.corpus_embeddings = []
        if corpus is not None:
            self.add_corpus(corpus)

    def __len__(self):
        """Get length of corpus."""
        return len(self.corpus)

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: {self.model}"
        if self.corpus:
            base += f", corpus size: {len(self.corpus)}"
        return base
    
    def to(self, device):
        self.model.to(device)
        self.device = device

    def add_corpus(self, corpus: Union[List[str], Dict[str, str]], batch_size: int = 32,
                   normalize_embeddings: bool = True, **kwargs):
        """
        使用文档chunk来转为向量
        :param corpus: 语料的list
        :param batch_size: batch size for computing embeddings
        :param normalize_embeddings: normalize embeddings before computing similarity
        :return: corpus, corpus embeddings
        """
        new_corpus = {}
        start_id = len(self.corpus) if self.corpus else 0
        for id, doc in enumerate(corpus):
            if isinstance(corpus, list):
                if doc not in self.corpus.values():
                    new_corpus[start_id + id] = doc
            else:
                if doc not in self.corpus.values():
                    new_corpus[id] = doc
        self.corpus.update(new_corpus)
        logger.info(f"Start computing corpus embeddings, new docs: {len(new_corpus)}")

        corpus_embeddings = self.encode(
            list(new_corpus.values()),
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize_embeddings,
            **kwargs
        ).tolist()

        if self.corpus_embeddings:
            self.corpus_embeddings = self.corpus_embeddings + corpus_embeddings
        else:
            self.corpus_embeddings = corpus_embeddings
        logger.info(f"Add {len(new_corpus)} docs, total: {len(self.corpus)}, emb len: {len(self.corpus_embeddings)}")

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
        """Returns the embeddings for a batch of sentences."""
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
    
    def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]], score_function:str="cos_sim", **kwargs):
        """
        Compute similarity between two texts.
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
        text_emb1 = self.encode(a, **kwargs)
        text_emb2 = self.encode(b, **kwargs)

        return score_function(text_emb1, text_emb2)

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """Compute cosine distance between two texts."""
        return 1 - self.similarity(a, b)

    def get_query_emb(self, queries, **kwargs):
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        queries_ids_map = {i: id for i, id in enumerate(list(queries.keys()))}
        queries_texts = list(queries.values())
        queries_embeddings = self.encode(queries_texts, convert_to_tensor=True, **kwargs)
        return queries, queries_embeddings, queries_ids_map
    
    def most_similar(self, queries: Union[str, List[str], Dict[str, str]], topk:int=10,
                     score_function:str="cos_sim", **kwargs):
        """
        Find the topk most similar texts to the queries against the corpus.
            It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.
        :param queries:str or list of str
        :param topk: int
        :param score_function: function to compute similarity, default cos_sim
        :param kwargs: additional arguments for the similarity function
        :return: Dict[str, Dict[str, float]], {query_id: {corpus_id: similarity_score}, ...}
        """

        queries, queries_embeddings, queries_ids_map = self.get_query_emb(queries, **kwargs)
        if score_function not in self.score_functions:
            raise ValueError(f"score function: {score_function} must be either (cos_sim) for cosine similarity"
                             " or (dot) for dot product")
        score_function = self.score_functions[score_function]
        corpus_embeddings = np.array(self.corpus_embeddings, dtype=np.float32)
        all_hits = semantic_search(queries_embeddings, corpus_embeddings, top_k=topk, score_function=score_function)
        
        result = {}
        for idx, hits in enumerate(all_hits):
            items = []
            for hit in hits[0:topk]:
                corpus_id = hit['corpus_id']
                items.append({**{'text': self.corpus[corpus_id]}, **hit})
            result[queries[queries_ids_map[idx]]] = items

        return result

    def save(self, corpus_path=None, emb_path=None):
        '''同时保存语料和embedding'''
        if corpus_path is not None:
            self.save_corpus(corpus_path)
        else:
            self.save_corpus()

        if emb_path is not None:
            self.save_embeddings(emb_path)
        else:
            self.save_embeddings()
    
    def load(self, corpus_path=None, emb_path=None):
        '''同时加载语料和embedding'''
        if corpus_path is not None:
            self.load_corpus(corpus_path)
        else:
            self.load_corpus()

        if emb_path is not None:
            self.load_embeddings(emb_path)
        else:
            self.load_embeddings()
    
    def save_corpus(self, corpus_path:str="corpus.txt"):
        with open(corpus_path, 'w', encoding='utf-8') as f:
            f.writelines(self.corpus)
    
    def load_corpus(self, corpus_path:str="corpus.txt"):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.corpus = f.readlines()

    def save_embeddings(self, emb_path:str="corpus_emb.json"):
        """
        Save corpus embeddings to json file.
        :param emb_path: json file path
        :return:
        """
        corpus_emb = {id: {"doc": self.corpus[id], "doc_emb": emb} for id, emb in
                      zip(self.corpus.keys(), self.corpus_embeddings)}
        with open(emb_path, "w", encoding="utf-8") as f:
            json.dump(corpus_emb, f, ensure_ascii=False)
        logger.debug(f"Save corpus embeddings to file: {emb_path}.")

    def load_embeddings(self, emb_path:str="corpus_emb.json"):
        """
        Load corpus embeddings from json file.
        :param emb_path: json file path
        :return: list of corpus embeddings, dict of corpus ids map, dict of corpus
        """
        try:
            with open(emb_path, "r", encoding="utf-8") as f:
                corpus_emb = json.load(f)
            corpus_embeddings = []
            for id, corpus_dict in corpus_emb.items():
                self.corpus[int(id)] = corpus_dict["doc"]
                corpus_embeddings.append(corpus_dict["doc_emb"])
            self.corpus_embeddings = corpus_embeddings
        except (IOError, json.JSONDecodeError):
            logger.error("Error: Could not load corpus embeddings from file.")