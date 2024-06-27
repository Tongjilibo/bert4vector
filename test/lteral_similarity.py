'''字面的相似度test
'''
from bert4vector.core import BertSimilarity, LongestCommonSubstringSimilarity
import pytest


def longest_common_substring_similarity():
    '''最长公共子序列相似度'''
    text2vec = LongestCommonSubstringSimilarity()
    sent1 = ['喜欢打篮球的男生喜欢什么样的女生', '西安下雪了？是不是很冷啊?', '第一次去见女朋友父母该如何表现？', '小蝌蚪找妈妈怎么样', 
            '给我推荐一款红色的车', '我喜欢北京', 'That is a happy person']
    sent2 = ['爱打篮球的男生喜欢什么样的女生', '西安的天气怎么样啊？还在下雪吗？', '第一次去见家长该怎么做', '小蝌蚪找妈妈是谁画的', 
            '给我推荐一款黑色的车', '我不喜欢北京', 'That is a happy dog']

    similarity = text2vec.similarity(sent1, sent2)
    print(similarity)

    text2vec.add_corpus(['你好', '我选你'])
    text2vec.add_corpus(['天气不错', '人很好看'])
    # text2vec.save(corpus_path='./corpus.json', emb_path='./emb.index')
    # text2vec.load(corpus_path='./corpus.json', emb_path='./emb.index')
    print(text2vec.search('你好'))
    print(text2vec.search(['你好', '天气晴']))


if __name__ == '__main__':
    longest_common_substring_similarity()