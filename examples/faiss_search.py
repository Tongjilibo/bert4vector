'''利用faiss进行语义检索
'''
from bert4vector.models import FaissVector

model = FaissVector('/data/pretrain_ckpt/simbert/sushen@simbert_chinese_tiny')

model.add_corpus(['你好', '我选你'], gpu_index=True)
model.add_corpus(['天气不错', '人很好看'], gpu_index=True)
print(model.most_similar('你好', topk=2))
print(model.most_similar(['你好', '天气晴']))