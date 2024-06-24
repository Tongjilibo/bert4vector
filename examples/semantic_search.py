from bert4vector import BertVector

model = BertVector('/data/pretrain_ckpt/simbert/sushen@simbert_chinese_tiny')

model.add_corpus(['你好', '我选你'])
model.add_corpus(['天气不错', '人很好看'])
print(model.search('你好'))
print(model.search(['你好', '天气晴']))