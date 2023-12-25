import pytest
from bert4vector import SentenceModel

emb_model_path = ['E:\pretrain_ckpt\embedding\BAAI@bge-large-en-v1.5', 
                  'E:\pretrain_ckpt\embedding\moka-ai@m3e-base', 
                  'E:/pretrain_ckpt/embedding/shibing624@text2vec-base-chinese']

@pytest.mark.parametrize("model_dir", emb_model_path)
def test_sentence_model(model_dir):
    model = SentenceModel(model_dir)
    query = ['天气不错', '我想去北京']
    embeddings = model.encode(query)
    print(embeddings)

if __name__ == '__main__':
    test_sentence_model('E:/pretrain_ckpt/embedding/shibing624@text2vec-base-chinese')
