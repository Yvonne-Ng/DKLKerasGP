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

#  Input data
from argparse import ArgumentParser
from h5py import File
import json
import uproot

# Drawing
from canvasEX import Canvas
from sklearn.externals import joblib
from numpy import ma

def parse_args():
    parser = ArgumentParser(description=__doc__)
    d = dict(help='%(default)s')
    parser.add_argument('input_file')
    parser.add_argument('signal_file', nargs='?')
    parser.add_argument('-e', '--output-file-extension', default='.pdf')
    parser.add_argument('-n', '--n-fits', type=int, default=5, **d)
    par_opts = parser.add_mutually_exclusive_group()
    par_opts.add_argument('-s', '--save-pars')
    par_opts.add_argument('-l', '--load-pars')

    parser.add_argument('-m', '--signal-events', type=float,
                          default=10000000, nargs='?', const=5)
    parser.add_argument('-g', "--guessTheMidPoints", type=bool, default=False)
    parser.add_argument('-t', "--halfPrediction", type=bool, default=False)
    parser.add_argument('-f', "--fix-hyperparams", type=bool, default=False)
    parser.add_argument('-b', "--bruteForceScan", type=bool, default=False)

    return parser.parse_args()

FIT_PARS = ['p0','p1','p2']
FIT_RANGE = (400, 1500)

def run():
    print("starting run")
    args = parse_args()
    ext = args.output_file_extension

    #---Getting background
    fBkg=uproot.open(args.input_file)['DSJ75yStar03_TriggerJets_J75_yStar03_mjj_TLA2016binning']
    #fBkg=uproot.open(args.input_file)['DSJ100yStar06_TriggerJets_J100_yStar06_mjj_TLA2016binning']
    x_bkg, y_bkg, xerr_bkg, yerr_bkg= get_xy_pts(fBkg, FIT_RANGE)
    print("xbkg: ", x_bkg)
    print("ybkg: ", y_bkg)


def get_xy_pts(f, x_range=None):
    vals, edges=f.numpy()
    #f.numpy()
    vals=np.asarray(vals)
    edges=np.array(edges)
    errors=np.sqrt(vals)
    center=(edges[:-1]+edges[1:])/2
    widths=np.diff(edges)

    if x_range is not None:
        low, high = x_range
        ok = (center > low) & (center < high)
    return center[ok], vals[ok], widths[ok], errors[ok]

if __name__ == '__main__':
    run()
