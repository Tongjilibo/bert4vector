from typing import List, Union, Dict, Literal
from .base import VectorSimilarity
from torch4keras.snippets import is_package_available
import numpy as np
from bert4vector.snippets import torch_Tensor


class BertSimilarity(VectorSimilarity):
    """ 在内存中存储和检索向量
    :param checkpoint_path: 模型权重地址
    :param config_path: 权重的config地址
    :param corpus: Corpus of documents to use for similarity queries.
    :param device: Device (like 'cuda' / 'cpu') to use for the computation.

    Example:
    ```python
    >>> from bert4vector.core import BertSimilarity
    >>> model = BertSimilarity('/data/pretrain_ckpt/simbert/sushen@simbert-chinese-tiny')
    >>> model.add_corpus(['你好', '我选你'], gpu_index=True)
    >>> model.add_corpus(['天气不错', '人很好看'], gpu_index=True)
    >>> print(model.search('你好', topk=2))
    >>> print(model.search(['你好', '天气晴']))

    >>> # {'你好': [{'corpus_id': 0, 'score': 0.9999, 'text': '你好'},
    ... #           {'corpus_id': 3, 'score': 0.5694, 'text': '人很好看'}]} 
    ```
    """
    def __init__(self, model_name_or_path:str, model_type:Literal['bert4torch', 'sentence_transformers']=None, 
                 corpus: List[str] = None, **model_config):
        self.model_type = model_type
        self.model = self.build_model(model_name_or_path, **model_config)
        super().__init__(corpus=corpus)
        self.matchint_type = 'BertSimilarity'
        self.emb_path = "corpus_emb.jsonl"

    def build_model(self, model_name_or_path, **model_config):
        '''初始化模型
        由于bert4torch支持的Embedding有限，因此也要兼容sentence_transformers，也可以手动显式指定model_type
        '''
        if self.model_type is None:
            if is_package_available('bert4torch'):
                # 有bert4torch包，且权重有bert4torch_config配置文件，表示是bert4torch支持的模型
                from bert4torch.pipelines import Text2Vec
                from bert4torch.snippets import get_config_path
                if get_config_path(model_name_or_path, allow_none=True):
                    return Text2Vec(model_name_or_path, **model_config)
            return self._get_sentence_transformers(model_name_or_path, **model_config)
        elif self.model_type == 'bert4torch':
            from bert4torch.pipelines import Text2Vec
            return Text2Vec(model_name_or_path, **model_config)
        elif self.model_type == 'sentence_transformers':
            return self._get_sentence_transformers(model_name_or_path, **model_config)
        else:
            raise ValueError(f'Args `model_type` {self.model_type} not supported')
    
    @staticmethod
    def _get_sentence_transformers(model_name_or_path, **model_config):
        '''从sentence_transformers加载'''
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name_or_path, **model_config)
    
    def to(self, device):
        self.model.to(device)
        self.device = device

    def encode(
            self,
            sentences: Union[str, List[str]],
            batch_size: int = 8,
            show_progress_bar: bool = False,
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            normalize_embeddings: bool = False,
            **kwargs
    ) -> Union[List[torch_Tensor], np.ndarray, torch_Tensor]:
        """ 把句子转换成向量
        Returns the embeddings for a batch of sentences.
        
        Example:
        ```python
        >>> from bert4vector.core import BertSimilarity
        >>> model = BertSimilarity('/data/pretrain_ckpt/simbert/sushen@simbert-chinese-tiny')   
        >>> sentences = ['喜欢打篮球的男生喜欢什么样的女生', '西安下雪了？是不是很冷啊?', 
        ...              '第一次去见女朋友父母该如何表现？', '小蝌蚪找妈妈怎么样', 
        ...              '给我推荐一款红色的车', '我喜欢北京']
        >>> vecs = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=False)
        >>> print(vecs.shape)
        >>> print(vecs)
        """
        return self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
            normalize_embeddings=normalize_embeddings,
            **kwargs
        )
