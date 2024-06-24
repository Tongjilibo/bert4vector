'''利用faiss进行语义检索
'''
from bert4vector.models import FaissVector

model = FaissVector('/data/pretrain_ckpt/simbert/sushen@simbert_chinese_tiny')

model.add_corpus(['你好', '我选你'], gpu_index=True)
model.add_corpus(['天气不错', '人很好看'], gpu_index=True)
model.save(corpus_path='./corpus.json', emb_path='./emb.index')
model.load(corpus_path='./corpus.json', emb_path='./emb.index')
print(model.search('你好', topk=2))
print(model.search(['你好', '天气晴']))