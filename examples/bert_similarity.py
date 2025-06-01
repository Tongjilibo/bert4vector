'''bert相似度
'''
from bert4vector.core import BertSimilarity


def encode(model_dir):
    model = BertSimilarity(model_dir)
    sentences = ['喜欢打篮球的男生喜欢什么样的女生', '西安下雪了？是不是很冷啊?', '第一次去见女朋友父母该如何表现？', '小蝌蚪找妈妈怎么样', 
                '给我推荐一款红色的车', '我喜欢北京']

    vecs = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=False)

    print(vecs.shape)
    print(vecs)

def search(model_dir):
    model = BertSimilarity(model_dir)
    model.add_corpus(['你好', '我选你'])
    model.add_corpus(['天气不错', '人很好看'])
    model.save(corpus_path='../cache/corpus.jsonl', emb_path='../cache/emb.index')
    model.load(corpus_path='../cache/corpus.jsonl', emb_path='../cache/emb.index')
    print(model.search('你好'))
    print(model.search(['你好', '天气晴'], return_dict=False))
    model.delete_corpus(['我选你', '人很好看'])

def similarity(model_dir):
    text2vec = BertSimilarity(model_dir)
    sent1 = ['喜欢打篮球的男生喜欢什么样的女生', '西安下雪了？是不是很冷啊?', '第一次去见女朋友父母该如何表现？', '小蝌蚪找妈妈怎么样', 
            '给我推荐一款红色的车', '我喜欢北京', 'That is a happy person']
    sent2 = ['爱打篮球的男生喜欢什么样的女生', '西安的天气怎么样啊？还在下雪吗？', '第一次去见家长该怎么做', '小蝌蚪找妈妈是谁画的', 
            '给我推荐一款黑色的车', '我不喜欢北京', 'That is a happy dog']

    similarity = text2vec.similarity(sent1, sent2)
    print(similarity)


if __name__ == '__main__':
    # model_dir = '/data/pretrain_ckpt/Tongjilibo/simbert_chinese_tiny'  # 调用bert4torch
    model_dir = '/data/pretrain_ckpt/maidalun1020/bce-embedding-base_v1'  # 调用sentence_transformers
    encode(model_dir)
    search(model_dir)
    similarity(model_dir)