''' 生成api server的示例代码
'''
from bert4vector.pipelines import EmbeddingSever

server = EmbeddingSever('/data/pretrain_ckpt/simbert/sushen@simbert_chinese_tiny')
server.run()