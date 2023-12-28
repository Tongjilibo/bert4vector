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
pip install bert4vecotr
```

- 安装最新版

```shell
pip install git+https://github.com/Tongjilibo/bert4vecotr
```

## 2. 支持的句向量权重
| 模型分类| 模型名称 | 权重来源| 权重链接 | 备注(若有)|
| ----- | ----- | ----- | ----- | ----- |
| simbert|simbert | 追一科技| [tf](https://github.com/ZhuiyiTechnology/simbert)，[torch_base](https://huggingface.co/peterchou/simbert-chinese-base/tree/main) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/simbert/convert_simbert.py) |
|        |simbert_v2/roformer-sim | 追一科技| [tf](https://github.com/ZhuiyiTechnology/roformer-sim)，[base](https://huggingface.co/junnyu/roformer_chinese_sim_char_base)，[ft_base](https://huggingface.co/junnyu/roformer_chinese_sim_char_ft_base)，[small](https://huggingface.co/junnyu/roformer_chinese_sim_char_small)，[ft_small](https://huggingface.co/junnyu/roformer_chinese_sim_char_ft_small)|[转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/simbert/convert_roformer-sim.py), [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/simbert) |
| embedding| text2vec-base-chinese |shibing624| [torch](https://huggingface.co/shibing624/text2vec-base-chinese) |[config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/embedding/moka-ai@m3e-base/bert4torch_config.json) |
|          | m3e |moka-ai| [torch](https://huggingface.co/moka-ai) |[config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/embedding/shibing624@text2vec-base-chinese/bert4torch_config.json)|
|          | bge |BAAI| [torch](huggingface.co) |[config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/embedding/moka-ai@m3e-base/bert4torch_config.json)|

## 3. 版本历史

|更新日期| bert4vector | bert4torch | 版本说明 |
|------| ---------------- | ----------------- |----------- |
|20231228| 0.0.2          | 0.4.4|初始版本，支持内存和faiss模式|

## 4. 更新历史：

- **20231228**：初始版本，支持内存和faiss模式


## 5. Reference
- [similarities](https://github.com/shibing624/similarities)
- [bert4vec](https://github.com/zejunwang1/bert4vec)