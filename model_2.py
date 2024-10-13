# include adaptive average pooling layers to accept images of variable length 

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import shelve

