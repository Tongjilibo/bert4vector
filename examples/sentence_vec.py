from bert4vector import Bert4Vector

model = Bert4Vector('E:/pretrain_ckpt/simbert/sushen@simbert_chinese_tiny')   
sentences = ['喜欢打篮球的男生喜欢什么样的女生', '西安下雪了？是不是很冷啊?', '第一次去见女朋友父母该如何表现？', '小蝌蚪找妈妈怎么样', 
             '给我推荐一款红色的车', '我喜欢北京']

vecs = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=False)

print(vecs.shape)
print(vecs)