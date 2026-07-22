__version__ = "1.0.0"

# preserved here for legacy reasons
__model_version__ = "latest"

import audiotools

audiotools.ml.BaseModel.INTERN += ["emac.**", "aar.**", "downstream.LatentMaskedTP"]
audiotools.ml.BaseModel.EXTERN += ["einops", "librosa", "transformers"]

from . import nn
from . import model
from . import utils
from .model import EMAC
from .model import EMACFile
from .model import AAR
from .model import SNACCodec
