'''调用openai兼容的embedding接口
'''

from bert4vector.core import OpenaiSimilarityRequest, OpenaiSimilarityAiohttp
import asyncio

url = "http://10.16.38.1:9901/v1/embeddings"
model_name = "bge-m3-Q4_k_M"
api_key = "gpustack_b1d636606794aa32_f1fd68bdd6ac9d6fd38a820bd9140664"

def openai_request():
    model = OpenaiSimilarityRequest(model_url=url, model_name=model_name, api_key=api_key)

    sentences = ['喜欢打篮球的男生喜欢什么样的女生', '西安下雪了？是不是很冷啊?', '第一次去见女朋友父母该如何表现？', '小蝌蚪找妈妈怎么样', 
                '给我推荐一款红色的车', '我喜欢北京']

    vecs = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=False)

    print(vecs.shape)
    print(vecs)


    # ========search========
    model.reset()
    model.add_corpus(['你好', '我选你'])
    model.add_corpus(['天气不错', '人很好看'])
    model.save(corpus_path='../cache/corpus.jsonl', emb_path='../cache/emb.index')
    model.load(corpus_path='../cache/corpus.jsonl', emb_path='../cache/emb.index')
    print(model.search('你好'))
    print(model.search(['你好', '天气晴'], return_dict=False))
    model.delete_corpus(['我选你', '人很好看'])


    # ========similarity========
    model.reset()
    sent1 = ['喜欢打篮球的男生喜欢什么样的女生', '西安下雪了？是不是很冷啊?', '第一次去见女朋友父母该如何表现？', '小蝌蚪找妈妈怎么样', 
            '给我推荐一款红色的车', '我喜欢北京', 'That is a happy person']
    sent2 = ['爱打篮球的男生喜欢什么样的女生', '西安的天气怎么样啊？还在下雪吗？', '第一次去见家长该怎么做', '小蝌蚪找妈妈是谁画的', 
            '给我推荐一款黑色的车', '我不喜欢北京', 'That is a happy dog']

    similarity = model.similarity(sent1, sent2)
    print(similarity)


async def openai_aiohttp():
    model = OpenaiSimilarityAiohttp(model_url=url, model_name=model_name, api_key=api_key)

    sentences = ['喜欢打篮球的男生喜欢什么样的女生', '西安下雪了？是不是很冷啊?', '第一次去见女朋友父母该如何表现？', '小蝌蚪找妈妈怎么样', 
                '给我推荐一款红色的车', '我喜欢北京']

    vecs = await model.encode(sentences, convert_to_numpy=True, normalize_embeddings=False)

    print(vecs.shape)
    print(vecs)


    # ========search========
    model.reset()
    await model.add_corpus(['你好', '我选你'])
    await model.add_corpus(['天气不错', '人很好看'])
    model.save(corpus_path='../cache/corpus.jsonl', emb_path='../cache/emb.index')
    model.load(corpus_path='../cache/corpus.jsonl', emb_path='../cache/emb.index')
    print(await model.search('你好'))
    print(await model.search(['你好', '天气晴'], return_dict=False))
    model.delete_corpus(['我选你', '人很好看'])


    # ========similarity========
    model.reset()
    sent1 = ['喜欢打篮球的男生喜欢什么样的女生', '西安下雪了？是不是很冷啊?', '第一次去见女朋友父母该如何表现？', '小蝌蚪找妈妈怎么样', 
            '给我推荐一款红色的车', '我喜欢北京', 'That is a happy person']
    sent2 = ['爱打篮球的男生喜欢什么样的女生', '西安的天气怎么样啊？还在下雪吗？', '第一次去见家长该怎么做', '小蝌蚪找妈妈是谁画的', 
            '给我推荐一款黑色的车', '我不喜欢北京', 'That is a happy dog']

    similarity = await model.similarity(sent1, sent2)
    print(similarity)


if __name__ == '__main__':
    openai_request()

    asyncio.run(openai_aiohttp())