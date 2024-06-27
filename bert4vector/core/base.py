from typing import List, Union, Dict
import json
from pathlib import Path
from loguru import logger
from torch4keras.snippets import print_table
import random


class Base:
    """基类
    """
    def __init__(self, corpus: List[str] = None, matching_type:str='Base') -> None:
        self.corpus = {}
        self.corpus_embeddings = {}
        self.matchint_type = matching_type
        if corpus is not None:
            self.add_corpus(corpus)

    def __len__(self):
        """Get length of corpus."""
        return sum([len(i) for i in self.corpus.values()])

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: {self.matchint_type}"
        if self.corpus:
            for k, v in self.corpus.items():
                base += f", sub_corpus={k}, data_size={len(v)}"  
            base += f", total size: {len(self)}"
        return base
    
    def reset(self, name:str=None):
        '''重置向量库'''
        if name is None:
            self.corpus = {}
            self.corpus_embeddings = {}
        elif name in self.corpus:
            self.corpus[name] = {}
            self.corpus_embeddings[name] = []
        else:
            logger.error(f'Args `name`={name} not in {list(self.corpus.keys())}')

    def summary(self, random_sample:bool=False, sample_count:int=2, verbose:int=1):
        '''统计一个各个sub_corpus的情况'''
        json_format, table_format = {}, []
        for name, sub_corpus in self.corpus.items():
            len_sub_corpus = len(sub_corpus)
            # 抽取少量样本
            if len_sub_corpus <= sample_count:
                smp_sub_corpus = list(sub_corpus.values())
            elif random_sample:
                smp_sub_corpus = random.sample(list(sub_corpus.values()), sample_count)
            else:
                smp_sub_corpus = []
                for v in sub_corpus.values():
                    if len(smp_sub_corpus) >= sample_count:
                        break
                    smp_sub_corpus.append(v)
            json_format[name] = {'size': len_sub_corpus, 'few_samples': smp_sub_corpus}
            table_format.append({**{'name': name}, **json_format[name]})
        
        if verbose != 0:
            logger.info('Corpus distribution statistics')
            print_table(table_format)
        return json_format
    
    def add_corpus(self, corpus: List[str], name:str='default'):
        """ 使用文档chunk来转为向量
        :param corpus: 语料的list
        :param name: sub_corpus名
        """
        new_corpus, new_corpus_set = {}, set()
        if name not in self.corpus:
            self.corpus[name] = {}
            self.corpus_embeddings[name] = []
        
        # 添加text到语料库
        id = len(self.corpus[name])
        sub_corpus_values = set(self.corpus[name].values())
        for doc in corpus:
            if (doc in sub_corpus_values) or (doc in new_corpus_set):
                continue
            new_corpus[id] = doc
            new_corpus_set.add(doc)
            id += 1

        self.corpus[name].update(new_corpus)
        del new_corpus_set
        return new_corpus

    def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """
        Compute similarity between two texts.
        :param a: list of str or str
        :param b: list of str or str
        :param score_function: function to compute similarity, default cos_sim
        :return: similarity score, torch.Tensor, Matrix with res[i][j] = cos_sim(a[i], b[j])
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """Compute cosine distance between two texts."""
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def search(self, queries: Union[str, List[str]], topk: int = 10):
        """ 在候选语料中寻找和query的向量最近似的topk个结果
        :param queries: Dict[str(query_id), str(query_text)] or List[str] or str
        :param topk: int
        :return: Dict[str, Dict[str, float]], {query_id: {corpus_id: similarity_score}, ...}
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")
    
    def save(self, corpus_path:Path=None, emb_path:Path=None):
        '''同时保存语料和embedding'''
        self._save_corpus(corpus_path)
        self._save_embeddings(emb_path)

    def load(self, corpus_path:Path=None, emb_path:Path=None):
        '''同时加载语料和embedding'''
        self._load_corpus(corpus_path)
        self._load_embeddings(emb_path)
    
    def _save_embeddings(index_path):
        pass

    def _load_embeddings(index_path):
        pass

    def _save_corpus(self, corpus_path:Path=None):
        '''保存语料到文件'''
        corpus_path = "corpus.json" if corpus_path is None else corpus_path
        with open(corpus_path, 'w', encoding='utf-8') as f:
            json.dump(self.corpus, f, ensure_ascii=False, indent=4)
        logger.info(f'Successfully save corpus: {corpus_path}')

    def _load_corpus(self, corpus_path:Path=None):
        '''从文件加载语料'''
        corpus_path = "corpus.json" if corpus_path is None else corpus_path
        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.corpus = json.load(f)
        # 修改id的type为int
        for name, sub_corpus in self.corpus.items():
            ids = list(sub_corpus.keys())
            for id in ids:
                sub_corpus[int(id)] = sub_corpus.pop(id)
        logger.info(f'Successfully load corpus: {corpus_path}')
