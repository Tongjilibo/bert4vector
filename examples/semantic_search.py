from bert4vector import Bert4Vector

model = Bert4Vector('E:/pretrain_ckpt/simbert/sushen@simbert_chinese_tiny')

sentences_path = "./sentences.txt"
model.build_index(sentences_path, ann_search=True, gpu_index=False, n_search=32)

results = model.search(queries=['一个男人在弹吉他。', '一个女人在做饭'], threshold=0.6, top_k=5)
# threshold为最低相似度阈值，top_k为查找的近邻个数
print(results)