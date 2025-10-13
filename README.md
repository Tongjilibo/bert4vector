# bert4vector

向量计算、存储、检索、相似度计算（兼容sentence_transformers）

[![licence](https://img.shields.io/github/license/Tongjilibo/bert4vector.svg?maxAge=3600)](https://github.com/Tongjilibo/bert4vector/blob/master/LICENSE)
[![GitHub release](https://img.shields.io/github/release/Tongjilibo/bert4vector.svg?maxAge=3600)](https://github.com/Tongjilibo/bert4vector/releases)
[![PyPI](https://img.shields.io/pypi/v/bert4vector?label=pypi%20package)](https://pypi.org/project/bert4vector/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/bert4vector)](https://pypistats.org/packages/bert4vector)
[![GitHub stars](https://img.shields.io/github/stars/Tongjilibo/bert4vector?style=social)](https://github.com/Tongjilibo/bert4vector)
[![GitHub Issues](https://img.shields.io/github/issues/Tongjilibo/bert4vector.svg)](https://github.com/Tongjilibo/bert4vector/issues)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Tongjilibo/bert4vector/issues)

[Documentation](https://bert4vector.readthedocs.io) |
[Bert4torch](https://github.com/Tongjilibo/bert4torch) |
[Examples](https://github.com/Tongjilibo/bert4vector/blob/master/examples) |
[Source code](https://github.com/Tongjilibo/bert4vector)

## 1. 下载安装

- 安装稳定版

```shell
pip install bert4vector
```

- 安装最新版

```shell
pip install git+https://github.com/Tongjilibo/bert4vector
```

## 2. 快速使用

- 向量计算

```python
from bert4vector.core import BertSimilarity
model = BertSimilarity('/data/pretrain_ckpt/Tongjilibo/simbert-chinese-tiny')
sentences = ['喜欢打篮球的男生喜欢什么样的女生', '西安下雪了？是不是很冷啊?', '第一次去见女朋友父母该如何表现？', '小蝌蚪找妈妈怎么样', '给我推荐一款红色的车', '我喜欢北京']
vecs = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=False)
print(vecs.shape)
# (6, 312)
```

- 相似度计算

```python
from bert4vector.core import BertSimilarity
text2vec = BertSimilarity('/data/pretrain_ckpt/Tongjilibo/simbert-chinese-tiny')
sent1 = ['你好', '天气不错']
sent2 = ['你好啊', '天气很好']
similarity = text2vec.similarity(sent1, sent2)
print(similarity)
# [[0.9075422  0.42991278]
#  [0.19584633 0.72635853]]
```

- 向量存储和检索

```python
from bert4vector.core import BertSimilarity
model = BertSimilarity('/data/pretrain_ckpt/Tongjilibo/simbert-chinese-tiny')
model.add_corpus(['你好', '我选你', '天气不错', '人很好看'])
print(model.search('你好'))
# {'你好': [{'corpus_id': 0, 'score': 0.9999, 'text': '你好'},
#           {'corpus_id': 3, 'score': 0.5694, 'text': '人很好看'}]} 
```

- api部署

```python
from bert4vector.pipelines import SimilaritySever
server = SimilaritySever('/data/pretrain_ckpt/embedding/BAAI--bge-base-zh-v1.5')
server.run(port=port)
# 接口调用可以参考'./examples/api.py'
```

## 3. 支持的句向量权重（除了以下权重，还支持 `sentence_transformers`支持的任意权重）

| 模型分类  | 模型名称                                                                 | 权重来源   | 权重链接                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | 备注(若有)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| --------- | ------------------------------------------------------------------------ | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| simbert   | [simbert](https://github.com/ZhuiyiTechnology/simbert)                      | 追一科技   | [`Tongjilibo/simbert-chinese-base`](https://huggingface.co/Tongjilibo/simbert-chinese-base)<br />[`Tongjilibo/simbert-chinese-small`](https://huggingface.co/Tongjilibo/simbert-chinese-small)<br />[`Tongjilibo/simbert-chinese-tiny`](https://huggingface.co/Tongjilibo/simbert-chinese-tiny)                                                                                                                                                                               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|           | [simbert_v2/roformer-sim](https://github.com/ZhuiyiTechnology/roformer-sim) | 追一科技   | [`junnyu/roformer_chinese_sim_char_base`](https://huggingface.co/junnyu/roformer_chinese_sim_char_base)<br />[`junnyu/roformer_chinese_sim_char_ft_base`](https://huggingface.co/junnyu/roformer_chinese_sim_char_ft_base)<br />[`junnyu/roformer_chinese_sim_char_small`](https://huggingface.co/junnyu/roformer_chinese_sim_char_small)<br />[`junnyu/roformer_chinese_sim_char_ft_small`](https://huggingface.co/junnyu/roformer_chinese_sim_char_ft_small)                 | [`junnyu/roformer_chinese_sim_char_base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/roformer_chinese_sim_char_base)<br />[`junnyu/roformer_chinese_sim_char_ft_base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/roformer_chinese_sim_char_ft_base)<br />[`junnyu/roformer_chinese_sim_char_small`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/roformer_chinese_sim_char_small)<br />[`junnyu/roformer_chinese_sim_char_ft_small`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/roformer_chinese_sim_char_ft_small)                                                                                             |
| embedding | [text2vec-base-chinese](https://github.com/shibing624/text2vec)             | shibing624 | [`shibing624/text2vec-base-chinese`](https://huggingface.co/shibing624/text2vec-base-chinese)                                                                                                                                                                                                                                                                                                                                                                               | [`shibing624/text2vec-base-chinese`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/text2vec-base-chinese)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|           | [m3e](https://github.com/wangyuxinwhy/uniem)                                | moka-ai    | [`moka-ai/m3e-base`](https://huggingface.co/moka-ai/m3e-base)                                                                                                                                                                                                                                                                                                                                                                                                               | [`moka-ai/m3e-base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/m3e-base)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|           | bge                                                                      | BAAI       | [`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5)<br />[`BAAI/bge-large-zh-v1.5`](https://huggingface.co/BAAI/bge-large-zh-v1.5)<br />[`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5)<br />[`BAAI/bge-base-zh-v1.5`](https://huggingface.co/BAAI/bge-base-zh-v1.5)<br />[`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5)<br />[`BAAI/bge-small-zh-v1.5`](https://huggingface.co/BAAI/bge-small-zh-v1.5) | [`BAAI/bge-large-en-v1.5`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bge-large-en-v1.5)<br />[`BAAI/bge-large-zh-v1.5`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bge-large-zh-v1.5)<br />[`BAAI/bge-base-en-v1.5`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bge-base-en-v1.5)<br />[`BAAI/bge-base-zh-v1.5`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bge-base-zh-v1.5)<br />[`BAAI/bge-small-en-v1.5`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bge-small-en-v1.5)<br />[`BAAI/bge-small-zh-v1.5`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bge-small-zh-v1.5) |
|           | gte                                                                      | thenlper   | [`thenlper/gte-large-zh`](https://huggingface.co/thenlper/gte-large-zh)<br />[`thenlper/gte-base-zh`](https://huggingface.co/thenlper/gte-base-zh)                                                                                                                                                                                                                                                                                                                           | [`thenlper/gte-base-zh`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/gte-base-zh)<br />[`thenlper/gte-large-zh`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/gte-large-zh)                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

*注：

1. 除了以上模型外，也支持 `sentence_transformers`支持的任意模型
2. `高亮格式`(如 `Tongjilibo/simbert-chinese-small`)的表示可直接联网下载
3. 国内镜像网站加速下载

   - `HF_ENDPOINT=https://hf-mirror.com python your_script.py`
   - `export HF_ENDPOINT=https://hf-mirror.com`后再执行python代码
   - 在python代码开头如下设置

   ```python
   import os
   os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
   ```

## 4. 版本历史

| 更新日期 | bert4vector | 版本说明                                                                                                       |
| -------- | ----------- | -------------------------------------------------------------------------------------------------------------- |
| 20251013 | 0.0.7.post2 | 去除对torch的完全依赖                                                                                          |
| 20251009 | 0.0.7       | 增加 `OpenaiSimilarityRequest`和 `OpenaiSimilarityAiohttp`用于访问openai格式的远程模型                     |
| 20250601 | 0.0.6       | `add_corpus`增加 `corpus_property`入参；增加 `delete_corpus`方法；支持任意 `sentence_transformers`模型 |
| 20240928 | 0.0.5       | 小修改，api中可以reset                                                                                         |
| 20240710 | 0.0.4       | 增加最长公共子序列字面召回，不安装torch也可以使用部分功能                                                      |
| 20240628 | 0.0.3       | 增加多种字面召回，增加api接口部署                                                                              |

## 5. 更新历史：

- **20240928**：小修改，api中可以reset
- **20240710**：增加最长公共子序列字面召回，不安装torch也可以使用部分功能
- **20240628**：增加多种字面召回，增加api接口部署

## 6. Reference

- [similarities](https://github.com/shibing624/similarities)
- [bert4vec](https://github.com/zejunwang1/bert4vec)
