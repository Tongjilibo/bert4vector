'''利用faiss进行语义检索
'''
from bert4vector.core import FaissSimilarity

model = FaissSimilarity('/data/pretrain_ckpt/Tongjilibo/simbert_chinese_tiny')

model.add_corpus(['你好', '我选你'], gpu_index=True)
model.add_corpus(['天气不错', '人很好看'], gpu_index=True)
model.summary()
model.save(corpus_path='../cache/corpus.jsonl', emb_path='../cache/emb.index')
model.load(corpus_path='../cache/corpus.jsonl', emb_path='../cache/emb.index')
print(model.search('你好', topk=3))
print(model.search(['你好', '天气晴'], return_dict=False))