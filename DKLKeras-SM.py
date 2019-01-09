""" test running of the TLA spectrum using the deep kernel learning spectral mixture kernel, basing from the example https://github.com/alshedivat/keras-gp/blob/master/examples/msgp_sm_kernel_mlp_kin40k.py"""

import os
import numpy as np
np.random.seed(42)

# Keras
from keras.layers import Input, Dense, Dropout
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping

# KGP
from kgp.models import Model
from kgp.layers import GP

# Dataset interfaces
from kgp.datasets.kin40k import load_data

# Model assembling and executing
from kgp.utils.experiment import train

# Metrics & losses
from kgp.losses import gen_gp_loss
from kgp.metrics import root_mean_squared_error as RMSE
