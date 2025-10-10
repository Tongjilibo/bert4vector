from typing import List, Union, Dict
import json
from pathlib import Path
from loguru import logger
from torch4keras.snippets import print_table
import random
from tqdm import tqdm
from bert4vector.snippets import cos_sim, dot_score, semantic_search
from pathlib import Path
import numpy as np
import os


SEARCH_RES_TYPE = Union[List, Dict[str, List]]  # search返回的数据类型


class SimilarityBase:
    """基类
    """
    def __init__(self, corpus: List[str] = None) -> None:
        self.corpus = {}
        self.corpus_embeddings = {}
        # self.matchint_type = 'Base'
        self.corpus_path = 'config.jsonl'
        self.emb_path = 'emg.jsonl'
        if corpus is not None:
            self.add_corpus(corpus)

    @property
    def matching_type(self):
        return self.__class__.__name__
    
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
        '''sentences转为编码'''
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def summary(self, random_sample:bool=False, sample_count:int=2, verbose:int=1):
        '''统计语料库分布的情况'''
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
    
    def delete_corpus(self, corpus:List[str]=None, corpus_ids:List[int]=None, name:str='default', **kwargs):
        '''按照corpus或者corpus_ids来删除部分语料，包含corpus语料和embedding
        :param corpus: 待删除语料的list
        :param corpus_ids: 待删除语料的id
        :param name: sub_corpus名
        '''
        assert corpus is not None or corpus_ids is not None, 'Args `corpus` and `corpus_ids` can not be None at the same time'

        if name not in self.corpus:
            logger.error(f'Args `name`={name} not in {list(self.corpus.keys())}')
            return
        
        delete_count = 0
        if corpus:
            corpus = set(corpus)
            del_id_set = set()
            for id, text in self.corpus[name].items():
                if text['text'] in corpus:
                    del_id_set.add(id)
                    delete_count += 1
        elif corpus_ids:
            del_id_set = set(corpus_ids)
            delete_count += len(del_id_set)

        sub_corpus = [(k,v) for k, v in self.corpus[name].items() if k not in del_id_set]
        sub_corpus = sorted(sub_corpus, key=lambda x: x[0])
        self.corpus[name] = {id:item for id, (_, item) in enumerate(sub_corpus)}
        self.corpus_embeddings[name] = [v for i, v in enumerate(self.corpus_embeddings[name]) if i not in del_id_set]
        logger.info(f'Successfuly delete {delete_count} docs from corpus `{name}`')
        return delete_count

    def _get_corpus_text(self, name:str='default') -> list:
        '''获取语料列表'''
        return [item['text'] for item in self.corpus[name].values()]

    def add_corpus(self, corpus: List[str], name:str='default', corpus_property:List[dict]=None, **kwargs):
        """ 使用文档chunk来转为向量
        :param corpus: 语料的list
        :param name: sub_corpus名
        :param corpus_property: 语料属性，由于corpus是去重时候会结果不同，为了防止和外部属性对应不上，这里可以返回
        """
        # 添加语料并放到语料库
        new_corpus = self._add_corpus(corpus=corpus, name=name, corpus_property=corpus_property)
        
        # 转向量并放到向量库
        self._add_embedding(corpus=[item['text'] for item in new_corpus.values()], name=name, **kwargs)

        # log
        msg = f"Add {len(new_corpus)} docs for `{name}`, total: {len(self.corpus[name])}"
        if len(self.corpus_embeddings[name]) > 0:
            msg += f", emb dim: {len(self.corpus_embeddings[name][0])}"
        logger.info(msg)

    def _add_corpus(self, corpus: List[str], name:str='default', corpus_property:List[dict]=None, **kwargs) -> Dict[int, dict]:
        '''添加语料并放到语料库'''
        assert isinstance(corpus, (list, tuple, set)) and all([isinstance(c, str) for c in corpus]), \
            'Args `corpus` only support List[str] format'
        if corpus_property:
            assert isinstance(corpus_property, (list, tuple, set)) and all([isinstance(c, dict) for c in corpus_property]), \
                'Args `corpus_property` only support List[dict] format'
            assert len(corpus)==len(corpus_property), \
                f'Not same length: len(corpus)={len(corpus)} <> len(corpus_property)={len(corpus_property)}'

        new_corpus, new_corpus_set = {}, set()
        if name not in self.corpus:
            self.corpus[name] = {}
            self.corpus_embeddings[name] = []
        
        # 添加text到语料库
        id = len(self.corpus[name])
        sub_corpus_values = self._get_corpus_text(name)
        for i, doc in enumerate(corpus):
            # 在存量corpus中，或在新的语料中，则不添加
            if (doc in sub_corpus_values) or (doc in new_corpus_set):
                continue
            new_corpus[id] = {'text': doc}
            # 更新属性
            if corpus_property:
                if 'text' in corpus_property[i]:  # 防止重名
                    corpus_property[i]['text_new'] = corpus_property[i].pop('text')
                new_corpus[id].update(corpus_property[i])
            new_corpus_set.add(doc)
            id += 1

        self.corpus[name].update(new_corpus)
        del new_corpus_set
        return new_corpus
    
    def _add_embedding(self, corpus:List[str], name:str='default', **kwargs):
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
        for sentence in tqdm(corpus, desc="Encoding"):
            sent_emb = self.encode(sentence, **kwargs)
            if isinstance(sent_emb, np.ndarray):
                sent_emb = sent_emb.tolist()                
            corpus_embeddings.append(sent_emb)
        self.corpus_embeddings[name] = self.corpus_embeddings.get(name, []) + corpus_embeddings

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
        os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
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

    @staticmethod
    def _get_search_result(result:Dict[str, list], return_dict:bool=True):
        '''返回字典还是列表'''
        if return_dict:
            return result
        return [i for i in result.values()]


class PairedSimilarity(SimilarityBase):
    '''成对的texts组相似度计算'''
    def __init__(self, corpus: List[str] = None):
        super().__init__(corpus=corpus)
        self.matchint_type = 'PairedSimilarity'
   
    def encode(self, sentences:Union[str, List], **kwargs):
        '''sentences转为编码, 这里默认不做任何处理'''
        return sentences

    def calc_pair_sim(self, emb1:str, emb2:str, **kwargs):
        '''计算两个编码之间的相似度'''
        raise NotImplementedError("cannot instantiate Abstract Base Class")
    
    def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]], **kwargs) -> List[float]:
        """计算两组texts之间的相同字符数占比相似度, 要求a和b的长度一致
        :param a:
        :param b:
        :return:
        """
        if isinstance(a, str):
            a = [a]
        if isinstance(b, str):
            b = [b]
        if len(a) != len(b):
            raise ValueError("expected two inputs of the same length")
        return [self.calc_pair_sim(self.encode(sentence1), self.encode(sentence2), **kwargs) 
                for sentence1, sentence2 in zip(a, b)]

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]) -> List[float]:
        """Compute cosine distance between two texts."""
        return [1 - s for s in self.similarity(a, b)]

    def search(self, queries: Union[str, List[str]], topk: int = 10, name:str='default', return_dict:bool=True) -> SEARCH_RES_TYPE:
        """Find the topn most similar texts to the query against the corpus."""
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]

        result = {}
        for query in queries:
            q_res = []
            query_emb = self.encode(query)
            for (corpus_id, text), doc_emb in zip(self.corpus[name].items(), self.corpus_embeddings[name]):
                score = self.similarity(query_emb, doc_emb)[0]
                q_res.append({**text, 'corpus_id': corpus_id, 'score': score})
            q_res.sort(key=lambda x: x['score'], reverse=True)
            result[query] = q_res[:topk]
        return self._get_search_result(result, return_dict)
    

class VectorSimilarity(SimilarityBase):
    '''基于向量的相似度计算'''
    def __init__(self, corpus: List[str] = None):
        super().__init__(corpus=corpus)
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}

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

    def search(self, queries: Union[str, List[str]], topk:int=10, score_function:str="cos_sim", name:str='default', return_dict:bool=True, **encode_kwargs) -> SEARCH_RES_TYPE:
        """ 在候选语料中寻找和query的向量最近似的topk个结果
        :param queries:str or list of str
        :param topk: int
        :param score_function: function to compute similarity, default cos_sim
        :param name: sub_corpus名
        :param encode_kwargs: additional arguments for the similarity function

        :return: Dict[str, Dict[str, float]], {query_id: {corpus_id: similarity_score}, ...}
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
                items.append({**self.corpus[name][corpus_id], **hit})
            result[queries[idx]] = items
        return self._get_search_result(result, return_dict)

    def _get_query_emb(self, queries:Union[str, List[str]], **encode_kwargs):
        '''获取query的句向量'''
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        queries_embeddings = self.encode(queries, **encode_kwargs)
        return queries, queries_embeddings

    def _add_embedding(self, corpus:List[str], name:str='default', **encode_kwargs):
        """ 使用文档chunk来转为向量

        >>> encode_kwargs参数
        :param batch_size: batch size for computing embeddings
        :param normalize_embeddings: normalize embeddings before computing similarity
        """
        # 转向量并放到向量库
        corpus_embeddings = self.encode(corpus, show_progress_bar=True, **encode_kwargs).tolist()
        self.corpus_embeddings[name] = self.corpus_embeddings.get(name, []) + corpus_embeddings


class AsyncVectorSimilarity(SimilarityBase):
    '''基于向量的相似度计算'''
    def __init__(self, corpus: List[str] = None):
        super().__init__(corpus=corpus)
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}

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
