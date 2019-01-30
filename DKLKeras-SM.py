""" test running of the TLA spectrum using the deep kernel learning spectral mixture kernel, basing from the example https://github.com/alshedivat/keras-gp/blob/master/examples/msgp_sm_kernel_mlp_kin40k.py"""

import os
import numpy as np
np.random.seed(42)

# Keras
import keras
from keras.layers import Input, Dense, Dropout
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

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

    #---Load Data
    fBkg=uproot.open(args.input_file)['DSJ75yStar03_TriggerJets_J75_yStar03_mjj_TLA2016binning']
    #fBkg=uproot.open(args.input_file)['DSJ100yStar06_TriggerJets_J100_yStar06_mjj_TLA2016binning']
    x_bkg, y_bkg, xerr_bkg, yerr_bkg= get_xy_pts(fBkg, FIT_RANGE)


    print("loading data ending")
    #--Setting Train Data and Test Data
    def oddElements(iList):
        return [iList[index] for index in range(len(iList)) if index%2==1]
    def evenElements(iList):
        return [iList[index] for index in range(len(iList)) if index%2==0]
    ##---- extracting Training set
    #X_train=evenElements(x_bkg)
    #y_train=evenElements(y_bkg)
    #yerr_train=evenElements(yerr_bkg)
    ##---- extracting prediction set
    #X_test=oddElements(x_bkg)
    #y_test=oddElements(y_bkg)
    #yerr_test=oddElements(yerr_bkg)

    def formatChange(l):
        """changing a list to a np array with an extra dimension"""
        l=np.array(l)
        output=l[:, np.newaxis]
        return output

    #---- extracting Training set
    #X_train=formatChange(evenElements(x_bkg))
    #y_train=formatChange(evenElements(y_bkg))
    # larger training set
    X_train=formatChange(x_bkg)
    y_train=formatChange(y_bkg)
    #yerr_train=formatChange(evenElements(yerr_bkg))
    #---- extracting prediction set
    X_test=formatChange(oddElements(x_bkg))
    y_test=formatChange(oddElements(y_bkg))
    #yerr_test=formatChange(oddElements(yerr_bkg))


    print("training set setup ending")

   # how is valid and test different in the original example?
    X_valid, y_valid = X_test, y_test

    data = {
	    'train': (X_train, y_train),
	    'valid': (X_valid, y_valid),
	    'test': (X_test, y_test),
	}
   #
# Model & training parameters
    input_shape = data['train'][0].shape[1:]
    print("input shape: ", input_shape)
    output_shape = data['train'][1].shape[1:]
#?
    batch_size = 2**10
    epochs = 500

    print("starting contstruction of model setup")
    # Construct & compile the model
    model = assemble_mlp(input_shape, output_shape, batch_size,
                         nb_train_samples=len(X_train))

    print("starting loss function  setup")
    loss = [gen_gp_loss(gp) for gp in model.output_layers]

    print("starting compiling")
    model.compile(optimizer=Adam(1e-4), loss=loss)
    #model.compile(optimizer=, loss=loss)

    print("got here")
    # Load saved weights (if exist)
    #if os.path.isfile('checkpoints/msgptrain_TLA.h5'):
    #    model.load_weights('checkpoints/msgptrain_TLA.h5', by_name=True)
    #    print("loading from previous check point")

    earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=0, mode='auto')


    # training

    history = train(model, data, callbacks=[earlyStopping,ModelCheckpoint(filepath='msgptrain_TLA', monitor='val_loss', save_best_only=True) ], gp_n_iter=5,epochs=epochs, batch_size=batch_size, verbose=1,checkpoint='msgptrain_TLA', checkpoint_monitor='val_mse')

    # Test the model
    X_test, y_test = data['test']
    y_preds = model.predict(X_test)
    rmse_predict = RMSE(y_test, y_preds)
    print('Test RMSE:', rmse_predict)


    #signif, _=res_significance(y_bkg, y_preds, x_bkg,t)

    with Canvas("fit"+".eps") as can:
        can.ax.set_yscale('log')
        can.ax.errorbar(X_test, y_test,  fmt='.', color="red", label="testing points")
        can.ax.errorbar(X_train, y_train, fmt='.', color="k", label="training points")
        can.ax.plot(X_test, np.squeeze(y_preds), '-r', label="GP Prediction")
        can.ax.legend(framealpha=0)
        can.ax.set_ylabel('events')
        #can.ratio.axhline(0, linewidth=1, alpha=0.5)
        #can.ratio.set_xlabel(r'$m_{jj}$ [GeV]', ha='right', x=0.98)
        #can.ratio.set_ylabel('significance')


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

def standardize_data(X_train, X_test, X_valid):
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)

    X_train -= X_mean
    X_train /= X_std
    X_test -= X_mean
    X_test /= X_std
    X_valid -= X_mean
    X_valid /= X_std

    return X_train, X_test, X_valid

def initCovSM(Q, D):
    print("initCovSM")
    w0 = np.log(np.ones((Q,1)))
    mu = np.log(np.maximum(0.05*np.random.rand(Q*D,1),1e-8))
    v = np.log(np.abs(np.random.randn(Q*D,1) + 1))

    print(" end of initCovSM")
    return [[w0], [mu], [v]]

def assemble_mlp(input_shape, output_shape, batch_size, nb_train_samples):
    """Assemble a simple MLP model.
    """
    print("assempble_mlp starting")
    print("inputshape:", input_shape)
    inputs = Input(shape=input_shape)
    hidden = Dense(200, activation='relu', name='dense1')(inputs)
    hidden = Dropout(0.5)(hidden)
    hidden = Dense(50, activation='relu', name='dense2')(hidden)
    hidden = Dropout(0.5)(hidden)
    hidden = Dense(25, activation='relu', name='dense3')(hidden)
    hidden = Dropout(0.25)(hidden)
    hidden = Dense(2, activation='relu', name='dense4')(hidden)

    print("NN set up completed")
    gp = GP(hyp={
                'lik': np.log(0.3),
                'mean': np.zeros((2,1)).tolist() + [[0]],
                'cov': initCovSM(6,1),
            },
            inf='infGrid', dlik='dlikGrid',
            opt={'cg_maxit': 2000, 'cg_tol': 1e-6},
            mean='meanSum', cov='covSM',
            update_grid=1,
            grid_kwargs={'eq': 1, 'k': 70.},
            cov_args=[6],
            mean_args=['{@meanLinear, @meanConst}'],
            batch_size=batch_size,
            nb_train_samples=nb_train_samples)
    outputs = [gp(hidden)]

    return Model(inputs=inputs, outputs=outputs)


if __name__ == '__main__':
    run()
