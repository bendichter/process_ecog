from __future__ import print_function, division

import argparse, h5py, time, os
import numpy as np

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


def preprocess_data(path, subject, blocks, bands='default', bands_vals=None):
    for block in blocks:
        block_path = os.path.join(path, '{}_B{}.nwb'.format(subject, block))
        transform(block_path, bands='default', bands_vals=None)


def transform(block_path, bands='default', bands_vals=None):
    """
    Takes raw LFP data and does the standard Hilbert algorithm:
    1) CAR
    2) notch filters
    3) Hilbert transform on different bands

    Takes about 20 minutes to run on 1 10-min block.

    Parameters
    ----------
    block_path : str
        subject file path
    bands: str, optional
        Frequency bands to filter the signal.
        'default' for Chang lab default values
        'high_gamma' for 70~150 Hz
        'custom' for user defined
    bands_vals: 2D array, necessary only if bands='custom'
        [2,nBands] numpy array with gaussian filter parameters, where:
        bands_vals[1,:] = filter centers
        bands_vals[2,:] = filter sigmas

    Returns
    -------
    Saves preprocessed signals (LFP) and spectral power (DecompositionSeries) in
    the current NWB file. Only if containers for these data do not exist in the
    file.
    """
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
        X_fft_h = None
        for ii, (cf, sd) in enumerate(zip(cfs, sds)):
             kernel = gaussian(X, rate, cf, sd)
             X_analytic, X_fft_h = hilbert_transform(X, rate, kernel, phase=None, X_fft_h=X_fft_h)
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
