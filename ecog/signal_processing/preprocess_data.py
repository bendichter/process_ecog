from __future__ import print_function, division

import argparse, h5py, time, os
import numpy as np
from scipy.io import loadmat

from ecog.signal_processing import resample
from ecog.signal_processing import subtract_CAR
from ecog.signal_processing import linenoise_notch
from ecog.signal_processing import hilbert_transform
from ecog.signal_processing import gaussian
from ecog.utils import load_bad_electrodes, bands

import nwbext_ecog
from pynwb import NWBHDF5IO, ProcessingModule
from pynwb.ecephys import LFP
from pynwb.core import DynamicTable, DynamicTableRegion, VectorData
from pynwb.misc import DecompositionSeries


def preprocess_data(path, subject, blocks, phase=False):
    for block in blocks:
        block_path = os.path.join(path, '{}_B{}.nwb'.format(subject, block))
        transform(block_path, phase=phase)


def transform(block_path, suffix=None, phase=False, seed=20180928):
    """
    Takes raw LFP data and does the standard hilb algorithm:
    1) CAR
    2) notch filters
    3) Hilbert transform on different bands
    ...

    Saves to os.path.join(block_path, subject + '_B' + block + '_AA.h5')

    Parameters
    ----------
    block_path
    rate
    cfs: filter center frequencies. If None, use Chang lab defaults
    sds: filer standard deviations. If None, use Chang lab defaults

    takes about 20 minutes to run on 1 10-min block
    """

    rng = None
    if phase:
        rng = np.random.RandomState(seed)
    rate = 400.

    cfs = bands.chang_lab['cfs']
    sds = bands.chang_lab['sds']

    subj_path, block_name = os.path.split(block_path)
    block_name = os.path.splitext(block_path)[0]

    start = time.time()

    with NWBHDF5IO(block_path, 'r+') as io:
        nwb = io.read()
        # 1e6 scaling helps with numerical accuracy
        X = nwb.acquisition['ECoG'].data[:].T * 1e6
        fs = nwb.acquisition['ECoG'].rate
        bad_elects = load_bad_electrodes(nwb)
        print('Load time for h5 {}: {} seconds'.format(block_name,
                                                       time.time() - start))
        print('rates {}: {} {}'.format(block_name, rate, fs))
        if not np.allclose(rate, fs):
            assert rate < fs
            X = resample(X, rate, fs)

        if bad_elects.sum() > 0:
            X[bad_elects] = np.nan

        # Subtract CAR
        start = time.time()
        X = subtract_CAR(X)
        print('CAR subtract time for {}: {} seconds'.format(block_name,
                                                            time.time()-start))

        # Apply Notch filters
        start = time.time()
        X = linenoise_notch(X, rate)
        print('Notch filter time for {}: {} seconds'.format(block_name,
                                                            time.time()-start))

        # Apply Hilbert transform
        X = X.astype('float32')   #signal (nChannels,nSamples)
        nChannels = X.shape[0]
        nSamples = X.shape[1]
        nBands = len(cfs)
        Xp = np.zeros((nBands, nChannels, nSamples))  #power (nBands,nChannels,nSamples)
        theta = None
        if phase:
             theta = rng.rand(*X.shape) * 2. * np.pi
             theta = np.sin(theta) + 1j * np.cos(theta)
        X_fft_h = None
        for ii, (cf, sd) in enumerate(zip(cfs, sds)):
             kernel = gaussian(X, rate, cf, sd)
             X_analytic, X_fft_h = hilbert_transform(X, rate, kernel, phase=theta, X_fft_h=X_fft_h)
             Xp[ii] = abs(X_analytic).astype('float32')

        # Scales signals back to Volt
        X /= 1e6

        # Save preprocessed data and power estimate at NWB file
        # Create LFP data interface container
        lfp = LFP()

        # Add preprocessed downsampled signals as an electrical_series
        elecs_region = nwb.electrodes.create_region(name='electrodes',
                                                    region=np.arange(nChannels).tolist(),
                                                    description='')
        lfp_ts = lfp.create_electrical_series(name='preprocessed',
                                              data=X,
                                              electrodes=elecs_region,
                                              rate=400.,
                                              description='')

        # Add spectral band power as a decomposition_series
        # bands: (DynamicTable) a table for describing the frequency bands that the signal was decomposed into
        cfsV = VectorData(name='filter_center',
                          description='frequencies for bandpass filters',
                          data=cfs)
        sdsV = VectorData(name='filter_sigma',
                          description='frequencies for bandpass filters',
                          data=sds)
        bandsTable = DynamicTable(name='bands',
                                  description='Series of filters used for Hilbert transform.',
                                  columns=[cfsV,sdsV],
                                  colnames=['filter_center','filter_sigma'])

        # data: (ndarray) dims: num_times * num_channels * num_bands
        Xp = np.swapaxes(Xp,0,2)
        decs = DecompositionSeries(name='Bandpower',
                                    data=Xp,
                                    description='Band power estimated with Hilbert transform.',
                                    metric='power',
                                    unit='V**2/Hz',
                                    bands=bandsTable,
                                    rate=400.,
                                    source_timeseries=lfp_ts)

        # Create ecephys ProcessingModule
        ecephys_module = ProcessingModule(name='ecephys',
                                          description='Extracellular electrophysiology data.')

        # Add LFP data interface container to ecephys ProcessingModule
        ecephys_module.add_data_interface(lfp)
        ecephys_module.add_data_interface(decs)

        # Add module to NWB file
        nwb.add_processing_module(ecephys_module)
        io.write(nwb)
