from loguru import logger
from typing import List, Union, Dict
import numpy as np
import json
import os
import torch
from bert4torch.models import build_transformer_model
from bert4torch.snippets import get_pool_emb, sequence_padding
from bert4torch.tokenizers import Tokenizer
from bert4vector.base import Base
from bert4vector.utils import cos_sim, dot_score, semantic_search


class SentenceModel:
    def __init__(self, model_path=None, vocab_path=None, config_path=None, checkpoint_path=None, device='cpu', model_config=None) -> None:
        if model_path is not None:
            vocab_path = vocab_path or os.path.join(model_path, 'vocab.txt')
            config_path = config_path or os.path.join(model_path, 'config.json')
            checkpoint_path = checkpoint_path or [i for i in os.listdir(model_path) if i.endswith('.bin')]
        self.tokenizer = Tokenizer(vocab_path, do_lower_case=True)
        model_config = model_config or dict()
        self.model = build_transformer_model(config_path, checkpoint_path, return_dict=True, **model_config).to(device)
        self.device = device
    
    def encode(
            self,
            sentences: Union[str, List[str]],
            batch_size: int = 8,
            pool_strategy='cls',
            custom_layer=None,
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            normalize_embeddings: bool = True,
            ):
        embeddings = []
        if isinstance(sentences, str):
            sentences = [sentences]
        for i in range(0, len(sentences) // batch_size):
            start = i * batch_size
            end = (i + 1) * batch_size
            batch = self.tokenizer(sentences[start:end])
            batch_input = [torch.tensor(sequence_padding(item), device=self.device) for item in batch]
            output = self.model(batch_input)

            last_hidden_state = output.get('last_hidden_state')
            pooler = output.get('pooled_output')
            attention_mask = (last_hidden_state != self.tokenizer._token_pad_id).long()
            embs = get_pool_emb(last_hidden_state, pooler, attention_mask, pool_strategy, custom_layer)
            embeddings.extend(embs)

        return embeddings
    

class BertVector(Base):
    def __init__(self, model_path, corpus: Union[List[str], Dict[str, str]] = None):
        """
        Initialize the similarity object.
        :param checkpoint_path: 模型权重地址
        :param config_path: 权重的config地址
        :param corpus: Corpus of documents to use for similarity queries.
        :param device: Device (like 'cuda' / 'cpu') to use for the computation.
        """
        self.model = SentenceModel(model_path)
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
    
    def add_corpus(self, corpus: Union[List[str], Dict[str, str]], batch_size: int = 32,
                   normalize_embeddings: bool = True):
        """
        使用文档chunk来转为向量
        :param corpus: 语料的碎片
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

        corpus_embeddings = self.get_embeddings(
            list(new_corpus.values()),
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize_embeddings,
        ).tolist()

        if self.corpus_embeddings:
            self.corpus_embeddings = self.corpus_embeddings + corpus_embeddings
        else:
            self.corpus_embeddings = corpus_embeddings
        logger.info(f"Add {len(new_corpus)} docs, total: {len(self.corpus)}, emb len: {len(self.corpus_embeddings)}")

    def get_embeddings(
            self,
            sentences: Union[str, List[str]],
            batch_size: int = 32,
            show_progress_bar: bool = False,
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            device: str = None,
            normalize_embeddings: bool = True,
    ):
        """Returns the embeddings for a batch of sentences."""
        if isinstance(sentences, str):
            sentences = [sentences]
        
        return self.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
            device=device,
            normalize_embeddings=normalize_embeddings,
        )
    
    def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]], score_function: str = "cos_sim", **kwargs):
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
        text_emb1 = self.get_embeddings(a, **kwargs)
        text_emb2 = self.get_embeddings(b, **kwargs)

        return score_function(text_emb1, text_emb2)

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """Compute cosine distance between two texts."""
        return 1 - self.similarity(a, b)

    def most_similar(self, queries: Union[str, List[str], Dict[str, str]], topn: int = 10,
                     score_function: str = "cos_sim", **kwargs):
        """
        Find the topn most similar texts to the queries against the corpus.
            It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.
        :param queries: str or list of str
        :param topn: int
        :param score_function: function to compute similarity, default cos_sim
        :param kwargs: additional arguments for the similarity function
        :return: Dict[str, Dict[str, float]], {query_id: {corpus_id: similarity_score}, ...}
        """
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        if score_function not in self.score_functions:
            raise ValueError(f"score function: {score_function} must be either (cos_sim) for cosine similarity"
                             " or (dot) for dot product")
        score_function = self.score_functions[score_function]
        result = {qid: {} for qid, query in queries.items()}
        queries_ids_map = {i: id for i, id in enumerate(list(queries.keys()))}
        queries_texts = list(queries.values())
        queries_embeddings = self.get_embeddings(queries_texts, convert_to_tensor=True, **kwargs)
        corpus_embeddings = np.array(self.corpus_embeddings, dtype=np.float32)
        all_hits = semantic_search(queries_embeddings, corpus_embeddings, top_k=topn, score_function=score_function)
        for idx, hits in enumerate(all_hits):
            for hit in hits[0:topn]:
                result[queries_ids_map[idx]][hit['corpus_id']] = hit['score']

        return result

    def save_embeddings(self, emb_path: str = "corpus_emb.json"):
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

    def load_embeddings(self, emb_path: str = "corpus_emb.json"):
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