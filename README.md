# bert4vector
向量计算、存储、检索、相似度计算


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
```python
from bert4vector import BertVector
model = BertVector('/data/pretrain_ckpt/simbert/sushen@simbert_chinese_tiny')
model.add_corpus(['你好', '我选你', '天气不错', '人很好看'], gpu_index=True)
print(model.search('你好', topk=2))
# {'你好': [{'corpus_id': 0, 'score': 0.9999, 'text': '你好'},
#           {'corpus_id': 3, 'score': 0.5694, 'text': '人很好看'}]} 
```
"""

## 3. 支持的句向量权重
| 模型分类| 模型名称 | 权重来源| 权重链接 | 备注(若有)|
| ----- | ----- | ----- | ----- | ----- |
| simbert|[simbert](https://github.com/ZhuiyiTechnology/simbert) | 追一科技|[`Tongjilibo/simbert-chinese-base`](https://huggingface.co/Tongjilibo/simbert-chinese-base), [`Tongjilibo/simbert-chinese-small`](https://huggingface.co/Tongjilibo/simbert-chinese-small), [`Tongjilibo/simbert-chinese-tiny`](https://huggingface.co/Tongjilibo/simbert-chinese-tiny) | |
|        |[simbert_v2/roformer-sim](https://github.com/ZhuiyiTechnology/roformer-sim) | 追一科技|[`junnyu/roformer_chinese_sim_char_base`](https://huggingface.co/junnyu/roformer_chinese_sim_char_base)，[`junnyu/roformer_chinese_sim_char_ft_base`](https://huggingface.co/junnyu/roformer_chinese_sim_char_ft_base)，[`junnyu/roformer_chinese_sim_char_small`](https://huggingface.co/junnyu/roformer_chinese_sim_char_small)，[`junnyu/roformer_chinese_sim_char_ft_small`](https://huggingface.co/junnyu/roformer_chinese_sim_char_ft_small)|[`roformer_chinese_sim_char_base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/roformer_chinese_sim_char_base), [`roformer_chinese_sim_char_ft_base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/roformer_chinese_sim_char_ft_base), [`roformer_chinese_sim_char_small`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/roformer_chinese_sim_char_small), [`roformer_chinese_sim_char_ft_small`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/roformer_chinese_sim_char_ft_small) |
| embedding| [text2vec-base-chinese](https://github.com/shibing624/text2vec) |shibing624| [`shibing624/text2vec-base-chinese`](https://huggingface.co/shibing624/text2vec-base-chinese) |[`text2vec-base-chinese`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/text2vec-base-chinese) |
|          | [m3e](https://github.com/wangyuxinwhy/uniem) |moka-ai| [`moka-ai/m3e-base`](https://huggingface.co/moka-ai/m3e-base) |[`m3e-base`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/m3e-base)|
|          | bge |BAAI| [`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5), [`BAAI/bge-large-zh-v1.5`](https://huggingface.co/BAAI/bge-large-zh-v1.5), [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5), [`BAAI/bge-base-zh-v1.5`](https://huggingface.co/BAAI/bge-base-zh-v1.5), [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5), [`BAAI/bge-small-zh-v1.5`](https://huggingface.co/BAAI/bge-small-zh-v1.5) | [`bge-large-en-v1.5`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bge-large-en-v1.5), [`bge-large-zh-v1.5`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bge-large-zh-v1.5), [`bge-base-en-v1.5`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bge-base-en-v1.5), [`bge-base-zh-v1.5`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bge-base-zh-v1.5), [`bge-small-en-v1.5`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bge-small-en-v1.5), [`bge-small-zh-v1.5`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/bge-small-zh-v1.5)|
|          | gte |thenlper| [`thenlper/gte-large-zh`](https://huggingface.co/thenlper/gte-large-zh), [`thenlper/gte-base-zh`](https://huggingface.co/thenlper/gte-base-zh) |[`gte-base-zh`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/gte-base-zh), [`gte-large-zh`](https://huggingface.co/Tongjilibo/bert4torch_config/tree/main/gte-large-zh)|

## 4. 版本历史

|更新日期| bert4vector | 版本说明 |
|------| ---------------- |----------- |
|20240628| 0.0.3   |增加多种字面召回，增加api接口部署|
|20240131| 0.0.2.post2   |去除对bert4torch的版本依赖|
|20231228| 0.0.2        |初始版本，支持内存和faiss模式|

## 5. 更新历史：

- **20240628**：增加多种字面召回，增加api接口部署
- **20231228**：初始版本，支持内存和faiss模式


## 6. Reference
- [similarities](https://github.com/shibing624/similarities)
- [bert4vec](https://github.com/zejunwang1/bert4vec)