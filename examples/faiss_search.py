from bert4vector import FaissVector

model = FaissVector('E:/pretrain_ckpt/simbert/sushen@simbert_chinese_tiny')

model.add_corpus(['你好', '我选你'], gpu_index=True)
model.add_corpus(['天气不错', '人很好看'], gpu_index=True)
print(model.search('你好'))
print(model.search(['你好', '天气晴']))