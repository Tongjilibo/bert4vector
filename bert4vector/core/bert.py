from typing import List, Union, Dict, Literal
from .base import VectorSimilarity


class BertSimilarity(VectorSimilarity):
    """ 在内存中存储和检索向量
    :param checkpoint_path: 模型权重地址
    :param config_path: 权重的config地址
    :param corpus: Corpus of documents to use for similarity queries.
    :param device: Device (like 'cuda' / 'cpu') to use for the computation.

    Example:
    ```python
    >>> from bert4vector import BertVector
    >>> model = BertVector('/data/pretrain_ckpt/simbert/sushen@simbert_chinese_tiny')
    >>> model.add_corpus(['你好', '我选你'], gpu_index=True)
    >>> model.add_corpus(['天气不错', '人很好看'], gpu_index=True)
    >>> print(model.search('你好', topk=2))
    >>> print(model.search(['你好', '天气晴']))

    >>> # {'你好': [{'corpus_id': 0, 'score': 0.9999, 'text': '你好'},
    ... #           {'corpus_id': 3, 'score': 0.5694, 'text': '人很好看'}]} 
    ```
    """
    def __init__(self, model_name_or_path:str, model_type:Literal['bert4torch', 'sentence_transformers']='bert4torch', 
                 corpus: List[str] = None, matching_type:str='BertSimilarity', **model_config):
        self.model_type = model_type
        self.model = self.build_model(model_name_or_path, **model_config)
        super().__init__(corpus=corpus, matching_type=matching_type)
        self.emb_path = "corpus_emb.jsonl"

    def build_model(self, model_name_or_path, **model_config):
        '''初始化模型'''
        if self.model_type == 'bert4torch':
            from bert4torch.pipelines import Text2Vec
            return Text2Vec(model_name_or_path, **model_config)
        elif self.model_type == 'sentence_transformers':
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(model_name_or_path, **model_config)
        else:
            raise ValueError(f'Args `model_type` {self.model_type} not supported')
    
    def to(self, device):
        self.model.to(device)
        self.device = device

    def encode(
            self,
            sentences: Union[str, List[str]],
            batch_size: int = 8,
            show_progress_bar: bool = False,
            pool_strategy=None,
            custom_layer=None,
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            normalize_embeddings: bool = False,
            max_seq_length: int = None
    ):
        """ 把句子转换成向量
        Returns the embeddings for a batch of sentences.
        
        Example:
        ```python
        >>> from bert4vector import BertVector
        >>> model = BertVector('/data/pretrain_ckpt/simbert/sushen@simbert_chinese_tiny')   
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
            pool_strategy=pool_strategy,
            custom_layer=custom_layer,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
            normalize_embeddings=normalize_embeddings,
            max_seq_length=max_seq_length
        )
    
    def _add_embedding(self, new_corpus: Dict[int, str], name:str='default', **encode_kwargs):
        """ 使用文档chunk来转为向量

        >>> encode_kwargs参数
        :param batch_size: batch size for computing embeddings
        :param normalize_embeddings: normalize embeddings before computing similarity
        """
        # 转向量并放到向量库
        corpus_embeddings = self.encode(
            list(new_corpus.values()),
            show_progress_bar=True,
            **encode_kwargs
        ).tolist()
        self.corpus_embeddings[name] = self.corpus_embeddings.get(name, []) + corpus_embeddings