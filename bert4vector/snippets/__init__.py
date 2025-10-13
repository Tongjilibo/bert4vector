from .distance import *
from .rank_bm25 import *
from .tfidf import *
from torch4keras.snippets import is_torch_available

if is_torch_available():
    from .util_torch import *
else:
    from .util_numpy import *
    class torch_Tensor: pass