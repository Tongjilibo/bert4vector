from typing import List, Union, Dict
import json
from pathlib import Path
from loguru import logger
from torch4keras.snippets import print_table
import random
from tqdm import tqdm


class SimilarityBase:
    """基类
    """
    def __init__(self, corpus: List[str] = None, matching_type:str='Base') -> None:
        self.corpus = {}
        self.corpus_embeddings = {}
        self.matchint_type = matching_type
        self.corpus_path = 'config.jsonl'
        self.emb_path = 'emg.jsonl'
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

    def encode(self, sentences:Union[str, List], **kwargs):
        '''sentences转为编码, 这里默认不做任何处理'''
        return sentences

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
    
    def add_corpus(self, corpus: List[str], name:str='default', **kwargs):
        """ 使用文档chunk来转为向量
        :param corpus: 语料的list
        :param name: sub_corpus名
        """
        # 添加语料并放到语料库
        new_corpus = self._add_corpus(corpus=corpus, name=name)
        
        # 转向量并放到向量库
        self._add_embedding(new_corpus=new_corpus, name=name, **kwargs)

        # log
        msg = f"Add {len(new_corpus)} docs for `{name}`, total: {len(self.corpus[name])}"
        if len(self.corpus_embeddings[name]) > 0:
            msg += f", emb dim: {len(self.corpus_embeddings[name][0])}"
        logger.info(msg)

    def _add_corpus(self, corpus: List[str], name:str='default', **kwargs):
        '''添加语料并放到语料库'''
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
    
    def _add_embedding(self, new_corpus:Dict[int, str], name:str='default', **kwargs):
        '''转向量并放到向量库
        :param corpus: 语料的list
        :param name: sub_corpus名
        '''
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str or dict of str
        """
        # 转向量并放到向量库
        corpus_embeddings = []
        for sentence in tqdm(list(new_corpus.values()), desc="Encoding"):
            corpus_embeddings.append(self.encode(sentence))
        self.corpus_embeddings[name] = self.corpus_embeddings.get(name) + corpus_embeddings

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
        logger.info(f'Successfully save corpus: {corpus_path or self.corpus_path}; emb: {emb_path or self.emb_path}')

    def load(self, corpus_path:Path=None, emb_path:Path=None):
        '''同时加载语料和embedding'''
        self._load_corpus(corpus_path)
        self._load_embeddings(emb_path)
        logger.info(f'Successfully load corpus: {corpus_path or self.corpus_path}; emb: {emb_path or self.emb_path}')
    
    def _save_embeddings(self, emb_path:Path=None):
        '''保存emb到文件'''
        emb_path = self.emb_path if emb_path is None else emb_path
        with open(emb_path, 'w', encoding='utf-8') as f:
            for name, sub_emb in self.corpus_embeddings.items():
                for emb in sub_emb:
                    json_obj = {"name": name, "emb": emb}
                    f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

    def _load_embeddings(self, emb_path:Path=None):
        '''从文件加载emb'''
        emb_path = self.emb_path if emb_path is None else emb_path
        self.corpus_embeddings = {}
        with open(emb_path, 'r', encoding='utf-8') as f:
            for line in f:
                json_obj = json.loads(line)
                name = json_obj['name']
                if name not in self.corpus_embeddings:
                    self.corpus_embeddings[name] = []
                self.corpus_embeddings[name].append(json_obj['emb'])

    def _save_corpus(self, corpus_path:Path=None):
        '''保存语料到文件'''
        corpus_path = self.corpus_path if corpus_path is None else corpus_path
        with open(corpus_path, 'w', encoding='utf-8') as f:
            for name, sub_corpus in self.corpus.items():
                for idx, text in sub_corpus.items():
                    json_obj = {"name": name, "id": idx, "text": text}
                    f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

    def _load_corpus(self, corpus_path:Path=None):
        '''从文件加载语料'''
        corpus_path = self.corpus_path if corpus_path is None else corpus_path
        self.corpus = {}
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                json_obj = json.loads(line)
                name = json_obj['name']
                if name not in self.corpus:
                    self.corpus[name] = {}
                self.corpus[name][json_obj['id']] = json_obj['text']
