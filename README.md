# bert4vector
向量计算、存储、检索、相似度计算


## 支持的句向量权重
| 模型分类| 模型名称 | 权重来源| 权重链接 | 备注(若有)|
| ----- | ----- | ----- | ----- | ----- |
| simbert|simbert | 追一科技| [tf](https://github.com/ZhuiyiTechnology/simbert)，[torch_base](https://huggingface.co/peterchou/simbert-chinese-base/tree/main) | [转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/simbert/convert_simbert.py) |
|        |simbert_v2/roformer-sim | 追一科技| [tf](https://github.com/ZhuiyiTechnology/roformer-sim)，[base](https://huggingface.co/junnyu/roformer_chinese_sim_char_base)，[ft_base](https://huggingface.co/junnyu/roformer_chinese_sim_char_ft_base)，[small](https://huggingface.co/junnyu/roformer_chinese_sim_char_small)，[ft_small](https://huggingface.co/junnyu/roformer_chinese_sim_char_ft_small)|[转换脚本](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/simbert/convert_roformer-sim.py), [config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/simbert) |
| embedding| text2vec-base-chinese |shibing624| [torch](https://huggingface.co/shibing624/text2vec-base-chinese) |[config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/embedding/moka-ai@m3e-base/bert4torch_config.json) |
|          | m3e |moka-ai| [torch](https://huggingface.co/moka-ai) |[config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/embedding/shibing624@text2vec-base-chinese/bert4torch_config.json)|
|          | bge |BAAI| [torch](huggingface.co) |[config](https://github.com/Tongjilibo/bert4torch/blob/master/examples/basic/embedding/moka-ai@m3e-base/bert4torch_config.json)|

## Reference
- [similarities](https://github.com/shibing624/similarities)
- [bert4vec](https://github.com/zejunwang1/bert4vec)