# -*- coding: utf-8 -*-
""" copy from https://github.com/shibing624/similarities/
"""

import json
import os
from typing import List, Union, Dict
import numpy as np
from loguru import logger
from .base import PairedSimilarity, VectorSimilarity
from ..snippets.distance import string_hash, hamming_distance, longest_common_substring_size, longest_common_subsequence_size
from ..snippets.rank_bm25 import BM25Okapi
from ..snippets.tfidf import TFIDF, load_stopwords, default_stopwords_file


pwd_path = os.path.abspath(os.path.dirname(__file__))

__all__ = [
    'SameCharsSimilarity',
    'LongestCommonSubstringSimilarity',
    'LongestCommonSubsequenceSimilarity',
    'HownetSimilarity',
    'SimHashSimilarity',
    'TfidfSimilarity',
    'BM25Similarity',
    'CilinSimilarity'
 ]


class SameCharsSimilarity(PairedSimilarity):
    """基于相同字符数占比计算相似度（不考虑文本字符位置顺序）

    ## Example:
    ```python
    >>> from bert4vector.core import SameCharsSimilarity
    >>> sent1 = ['喜欢打篮球的男生喜欢什么样的女生', '西安下雪了？是不是很冷啊?', '第一次去见女朋友父母该如何表现？', '小蝌蚪找妈妈怎么样', 
    ...         '给我推荐一款红色的车', '我喜欢北京', 'That is a happy person']
    >>> sent2 = ['爱打篮球的男生喜欢什么样的女生', '西安的天气怎么样啊？还在下雪吗？', '第一次去见家长该怎么做', '小蝌蚪找妈妈是谁画的', 
    ...         '给我推荐一款黑色的车', '我不喜欢北京', 'That is a happy dog']
    >>> text2vec = SameCharsSimilarity()
    >>> similarity = text2vec.similarity(sent1, sent2)
    >>> print(similarity)
  
    >>> text2vec.add_corpus(['你好', '我选你'])
    >>> text2vec.add_corpus(['天气不错', '人很好看'])
    >>> # text2vec.save(corpus_path='./corpus.json', emb_path='./emb.index')
    >>> # text2vec.load(corpus_path='./corpus.json', emb_path='./emb.index')
    >>> print(text2vec.search('你好'))
    >>> print(text2vec.search(['你好', '天气晴']))
    ```
    """
    def __init__(self, corpus: List[str] = None):
        super().__init__(corpus=corpus)

    def calc_pair_sim(self, emb1:str, emb2:str, **kwargs):
        '''计算两个编码之间的相似度'''
        if not emb1 or not emb2:
            return 0.0
        same = set(emb1) & set(emb2)
        similarity_score = max(len(same) / len(set(emb1)), len(same) / len(set(emb2)))
        return similarity_score


class LongestCommonSubstringSimilarity(PairedSimilarity):
    """基于最长公共子串占比计算相似度

    ## Example:
    ```python
    >>> from bert4vector.core import LongestCommonSubstringSimilarity
    >>> sent1 = ['喜欢打篮球的男生喜欢什么样的女生', '西安下雪了？是不是很冷啊?', '第一次去见女朋友父母该如何表现？', '小蝌蚪找妈妈怎么样', 
    ...         '给我推荐一款红色的车', '我喜欢北京', 'That is a happy person']
    >>> sent2 = ['爱打篮球的男生喜欢什么样的女生', '西安的天气怎么样啊？还在下雪吗？', '第一次去见家长该怎么做', '小蝌蚪找妈妈是谁画的', 
    ...         '给我推荐一款黑色的车', '我不喜欢北京', 'That is a happy dog']
    >>> text2vec = LongestCommonSubstringSimilarity()
    >>> similarity = text2vec.similarity(sent1, sent2)
    >>> print(similarity)
 
    >>> text2vec.add_corpus(['你好', '我选你'])
    >>> text2vec.add_corpus(['天气不错', '人很好看'])
    >>> # text2vec.save(corpus_path='./corpus.json', emb_path='./emb.index')
    >>> # text2vec.load(corpus_path='./corpus.json', emb_path='./emb.index')
    >>> print(text2vec.search('你好'))
    >>> print(text2vec.search(['你好', '天气晴']))
    ```
    """
    def __init__(self, corpus: List[str] = None,
                 min_same_len: int = 70, 
                 min_same_len_score: float = 0.9):
        super().__init__(corpus=corpus)
        self.min_same_len = min_same_len
        self.min_same_len_score = min_same_len_score

    def calc_pair_sim(self, emb1:str, emb2:str, **kwargs):
        if not emb1 or not emb2:
            return 0.0
        same_size = longest_common_substring_size(emb1, emb2)
        same_score = self.min_same_len_score if same_size > self.min_same_len else 0.0
        # 取最长公共子串/多个序列长度的最大值
        similarity_score = max(same_size / len(emb1), same_size / len(emb2), same_score)
        return similarity_score
    

class LongestCommonSubsequenceSimilarity(PairedSimilarity):
    """基于最长公共子串占比计算相似度
    """
    def __init__(self, corpus: List[str] = None,
                 min_same_len: int = 70, 
                 min_same_len_score: float = 0.9):
        super().__init__(corpus=corpus)
        self.min_same_len = min_same_len
        self.min_same_len_score = min_same_len_score

    def calc_pair_sim(self, emb1:str, emb2:str, **kwargs):
        if not emb1 or not emb2:
            return 0.0
        same_size = longest_common_subsequence_size(emb1, emb2)
        same_score = self.min_same_len_score if same_size > self.min_same_len else 0.0
        # 取最长公共子串/多个序列长度的最大值
        similarity_score = max(same_size / len(emb1), same_size / len(emb2), same_score)
        return similarity_score
    

class HownetSimilarity(PairedSimilarity):
    """计算两组texts之间的Hownet相似度
    """
    default_hownet_path = os.path.join(os.path.dirname(pwd_path), 'config/hownet.txt')

    def __init__(self, corpus: List[str] = None, hownet_path: str = default_hownet_path):
        # 加载Hownet语义词典
        self.hownet_dict = {}
        for line in open(hownet_path, 'r', encoding='utf-8'):
            words = [word for word in line.strip().replace(' ', '>').replace('\t', '>').split('>') if word != '']
            word = words[0]
            word_def = words[2]
            self.hownet_dict[word] = word_def.split(',')

        import jieba, jieba.posseg, jieba.analyse
        self.jieba, self.jieba.posseg, self.jieba.analyse = jieba, jieba.posseg, jieba.analyse
        super().__init__(corpus=corpus)

    def _word_sim(self, word1, word2):
        """比较两个词语之间的相似度"""
        sems_word1 = self.hownet_dict.get(word1, [])
        sems_words = self.hownet_dict.get(word2, [])
        scores = []
        for sem_word1 in sems_word1:
            for sem_word2 in sems_words:
                sem_inter = set(sem_word1).intersection(set(sem_word2))
                sem_union = set(sem_word1).union(set(sem_word2))
                scores.append(float(len(sem_inter)) / float(len(sem_union)))
        return max(scores) if scores else 0

    def calc_pair_sim(self, emb1:str, emb2:str):
        words1 = [word.word for word in self.jieba.posseg.cut(emb1) if word.flag[0] not in ['u', 'x', 'w']]
        words2 = [word.word for word in self.jieba.posseg.cut(emb2) if word.flag[0] not in ['u', 'x', 'w']]
        score_words1 = []
        score_words2 = []
        for word1 in words1:
            score = max(self._word_sim(word1, word2) for word2 in words2)
            score_words1.append(score)
        for word2 in words2:
            score = max(self._word_sim(word2, word1) for word1 in words1)
            score_words2.append(score)
        similarity_score = max(sum(score_words1) / len(words1), sum(score_words2) / len(words2))
        return similarity_score


class SimHashSimilarity(PairedSimilarity):
    """计算两组texts之间的SimHash相似度
    """
    def __init__(self, corpus: List[str] = None):
        super().__init__(corpus=corpus)
        self.emb_path = "hash_emb.jsonl"
        import jieba, jieba.posseg, jieba.analyse
        self.jieba, self.jieba.posseg, self.jieba.analyse = jieba, jieba.posseg, jieba.analyse

    def encode(self, sentences: Union[str, List[str]]):
        """
        Compute SimHash for a given text.
        :param sentence: str
        :return: hash code
        """
        if isinstance(sentences, str):
            is_input_string = True
            sentences = [sentences]
        else:
            is_input_string = False

        hash_codes = []
        for sentence in sentences:
            seg = self.jieba.cut(sentence)
            key_word = self.jieba.analyse.extract_tags('|'.join(seg), topK=None, withWeight=True, allowPOS=())
            # 先按照权重排序，再按照词排序
            key_list = []
            for feature, weight in key_word:
                weight = int(weight * 20)
                temp = []
                for f in string_hash(feature):
                    if f == '1':
                        temp.append(weight)
                    else:
                        temp.append(-weight)
                key_list.append(temp)
            content_list = np.sum(np.array(key_list), axis=0)
            # 编码读不出来
            if len(key_list) == 0:
                return '00'
            hash_code = ''
            for c in content_list:
                if c > 0:
                    hash_code = hash_code + '1'
                else:
                    hash_code = hash_code + '0'
            hash_codes.append(hash_code)
        return hash_codes[0] if is_input_string else hash_codes

    def calc_pair_sim(self, emb1:str, emb2:str):
        """Convert hamming distance to similarity score."""
        # 将距离转化为相似度
        score = 0.0
        if len(emb1) > 2 and len(emb2) > 2:
            score = 1 - hamming_distance(emb1, emb2, normalize=True)
        return score


class TfidfSimilarity(VectorSimilarity):
    """计算两组texts之间的Tfidf相似度
    """
    def __init__(self, corpus: List[str] = None):
        super().__init__(corpus=corpus)
        self.tfidf = TFIDF()

    def encode(self, sentences:Union[str, List], **kwargs) -> np.ndarray:
        if isinstance(sentences, List):
            emb = [self.tfidf.get_tfidf(sentence) for sentence in sentences]
        else:
            emb = self.tfidf.get_tfidf(sentences)
        return np.array(emb, dtype=np.float32)


class BM25Similarity(PairedSimilarity):
    """计算两组texts之间的BM25OKapi相似度
    目前只能召回, 并不能encode和similariry
    """
    def __init__(self, corpus: List[str] = None):
        super().__init__(corpus=corpus)
        self.bm25 = dict()
        self.default_stopwords = load_stopwords(default_stopwords_file)
        import jieba, jieba.posseg, jieba.analyse
        self.jieba, self.jieba.posseg, self.jieba.analyse = jieba, jieba.posseg, jieba.analyse

    def calc_pair_sim(self, emb1:str, emb2:str, **kwargs):
        raise NotImplementedError('BM25Similarity does not support similarity, please use search method instead.')
    
    def _add_embedding(self, corpus:List[str], name:str='default', **kwargs):
        """build bm25 model."""
        corpus_seg = [self.jieba.lcut(d) for d in corpus]
        corpus_seg = [[w for w in doc if (w.strip().lower() not in self.default_stopwords) and
                       len(w.strip()) > 0] for doc in corpus_seg]
        self.bm25[name] = BM25Okapi(corpus_seg)

    def search(self, queries: Union[str, List[str]], topk: int = 10, name:str='default', return_dict:bool=True) -> Dict[str, List]:
        if name not in self.bm25:
            raise ValueError(f'No bm25 model found for name: {name}')
        
        if isinstance(queries, str):
            queries = [queries]
        result = {}
        for query in queries:
            tokens = self.jieba.lcut(query)
            scores = self.bm25[name].get_scores(tokens)
            q_res = [{**self.corpus[name][corpus_id], 'corpus_id': corpus_id, 'score': score} for corpus_id, score in enumerate(scores)]
            q_res = sorted(q_res, key=lambda x: x['score'], reverse=True)[:topk]
            result[query] = q_res
        return self._get_search_result(result, return_dict)


class CilinSimilarity(PairedSimilarity):
    """ 计算两组texts之间的Cilin相似度
    """
    default_cilin_path = os.path.join(os.path.dirname(pwd_path), 'config/cilin.txt')

    def __init__(self, corpus: List[str] = None, cilin_path: str = default_cilin_path):
        super().__init__(corpus=corpus)
        self.cilin_dict = self.load_cilin_dict(cilin_path)  # Cilin(词林) semantic dictionary
        import jieba, jieba.posseg, jieba.analyse
        self.jieba, self.jieba.posseg, self.jieba.analyse = jieba, jieba.posseg, jieba.analyse

    @staticmethod
    def load_cilin_dict(path):
        """加载词林语义词典"""
        sem_dict = {}
        for line in open(path, 'r', encoding='utf-8'):
            line = line.strip()
            terms = line.split(' ')
            sem_type = terms[0]
            words = terms[1:]
            for word in words:
                if word not in sem_dict:
                    sem_dict[word] = sem_type
                else:
                    sem_dict[word] += ';' + sem_type

        for word, sem_type in sem_dict.items():
            sem_dict[word] = sem_type.split(';')
        return sem_dict

    def _word_sim(self, word1, word2):
        """
        比较计算词语之间的相似度，取max最大值
        :param word1:
        :param word2:
        :return:
        """
        sems_word1 = self.cilin_dict.get(word1, [])
        sems_word2 = self.cilin_dict.get(word2, [])
        score_list = []
        for sem_word1 in sems_word1:
            for sem_word2 in sems_word2:
                # 基于语义计算词语相似度
                sem1 = [sem_word1[0], sem_word1[1], sem_word1[2:4], sem_word1[4], sem_word1[5:7], sem_word1[-1]]
                sem2 = [sem_word2[0], sem_word2[1], sem_word2[2:4], sem_word2[4], sem_word2[5:7], sem_word2[-1]]
                score = 0
                for index in range(len(sem1)):
                    if sem1[index] == sem2[index]:
                        if index in [0, 1]:
                            score += 3
                        elif index == 2:
                            score += 2
                        elif index in [3, 4]:
                            score += 1
                score_list.append(score / 10)
        return max(score_list) if score_list else 0       
     
    def calc_pair_sim(self, sentence1:str, sentence2:str):
        words1 = [word.word for word in self.jieba.posseg.cut(sentence1) if word.flag[0] not in ['u', 'x', 'w']]
        words2 = [word.word for word in self.jieba.posseg.cut(sentence2) if word.flag[0] not in ['u', 'x', 'w']]
        score_words1 = []
        score_words2 = []
        for word1 in words1:
            score = max(self._word_sim(word1, word2) for word2 in words2)
            score_words1.append(score)
        for word2 in words2:
            score = max(self._word_sim(word2, word1) for word1 in words1)
            score_words2.append(score)
        similarity_score = max(sum(score_words1) / len(words1), sum(score_words2) / len(words2))
        return similarity_score