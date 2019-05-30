from __future__ import print_function, division

import argparse, h5py, time, os
import numpy as np

from ecog.signal_processing import resample
from ecog.signal_processing import subtract_CAR
from ecog.signal_processing import linenoise_notch
from ecog.signal_processing import hilbert_transform
from ecog.signal_processing import gaussian, hamming
from ecog.utils import load_bad_electrodes, bands

import nwbext_ecog
from pynwb import NWBHDF5IO, ProcessingModule
from pynwb.ecephys import LFP
from pynwb.core import DynamicTable, DynamicTableRegion, VectorData
from pynwb.misc import DecompositionSeries


def preprocess_data(path, subject, blocks, filter='default', bands_vals=None):
    for block in blocks:
        block_path = os.path.join(path, '{}_B{}.nwb'.format(subject, block))
        transform(block_path, filter=filter, bands_vals=bands_vals)


def transform(block_path, filter='default', bands_vals=None):
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
    filter: str, optional
        Frequency bands to filter the signal.
        'default' for Chang lab default values (Gaussian filters)
        'custom' for user defined (Gaussian filters)
    bands_vals: 2D array, necessary only if filter='custom'
        [2,nBands] numpy array with Gaussian filter parameters, where:
        bands_vals[0,:] = filter centers [Hz]
        bands_vals[1,:] = filter sigmas [Hz]

    Returns
    -------
    Saves preprocessed signals (LFP) and spectral power (DecompositionSeries) in
    the current NWB file. Only if containers for these data do not exist in the
    file.
    """
    write_file = 1
    rate = 400.

    # Define filter parameters
    if filter=='default':
        band_param_0 = bands.chang_lab['cfs']
        band_param_1 = bands.chang_lab['sds']
    elif filter=='high_gamma':
        band_param_0 = bands.chang_lab['cfs'][(bands.chang_lab['cfs'] > 70) & (bands.chang_lab['cfs'] < 150)]
        band_param_1 = bands.chang_lab['sds'][(bands.chang_lab['cfs'] > 70) & (bands.chang_lab['cfs'] < 150)]
        #band_param_0 = [ bands.neuro['min_freqs'][-1] ]  #for hamming window filter
        #band_param_1 = [ bands.neuro['max_freqs'][-1] ]
        #band_param_0 = bands.chang_lab['cfs'][29:37]      #for average of gaussian filters
        #band_param_1 = bands.chang_lab['sds'][29:37]
    elif filter=='custom':
        band_param_0 = bands_vals[0,:]
        band_param_1 = bands_vals[1,:]


    block_name = os.path.splitext(block_path)[0]

    start = time.time()

    with NWBHDF5IO(block_path, 'a') as io:
        nwb = io.read()

        # Storage of processed signals on NWB file -----------------------------
        if 'ecephys' not in nwb.modules:
            # Add module to NWB file
            nwb.create_processing_module(name='ecephys', description='Extracellular electrophysiology data.')
        ecephys_module = nwb.modules['ecephys']


        # LFP: Downsampled and power line signal removed
        if 'LFP' in nwb.modules['ecephys'].data_interfaces:
            lfp_ts = nwb.modules['ecephys'].data_interfaces['LFP'].electrical_series['preprocessed']
            X = lfp_ts.data[:].T
            rate = lfp_ts.rate
        else:

            # 1e6 scaling helps with numerical accuracy
            X = nwb.acquisition['ECoG'].data[:].T * 1e6
            fs = nwb.acquisition['ECoG'].rate
            bad_elects = load_bad_electrodes(nwb)
            print('Load time for h5 {}: {} seconds'.format(block_name,
                                                           time.time() - start))
            print('rates {}: {} {}'.format(block_name, rate, fs))
            if not np.allclose(rate, fs):
                assert rate < fs
                start = time.time()
                X = resample(X, rate, fs)
                print('resample time for {}: {} seconds'.format(block_name, time.time() - start))

            if bad_elects.sum() > 0:
                X[bad_elects] = np.nan

            # Subtract CAR
            start = time.time()
            X = subtract_CAR(X)
            print('CAR subtract time for {}: {} seconds'.format(block_name, time.time() - start))

            # Apply Notch filters
            start = time.time()
            X = linenoise_notch(X, rate)
            print('Notch filter time for {}: {} seconds'.format(block_name, time.time() - start))


            lfp = LFP()
            # Add preprocessed downsampled signals as an electrical_series
            lfp_ts = lfp.create_electrical_series(name='preprocessed',
                                                  data=X.T,
                                                  electrodes=nwb.acquisition['ECoG'].electrodes,
                                                  rate=rate,
                                                  description='')
            ecephys_module.add_data_interface(lfp)


        # Spectral band power
        if 'Bandpower_'+filter  not in nwb.modules['ecephys'].data_interfaces:

            # Apply Hilbert transform
            X = X.astype('float32')  # signal (nChannels,nSamples)
            nChannels = X.shape[0]
            nSamples = X.shape[1]
            nBands = len(band_param_0)
            Xp = np.zeros((nBands, nChannels, nSamples))  # power (nBands,nChannels,nSamples)
            X_fft_h = None
            for ii, (bp0, bp1) in enumerate(zip(band_param_0, band_param_1)):
                # if filter=='high_gamma':
                #    kernel = hamming(X, rate, bp0, bp1)
                # else:
                kernel = gaussian(X, rate, bp0, bp1)
                X_analytic, X_fft_h = hilbert_transform(X, rate, kernel, phase=None, X_fft_h=X_fft_h)
                Xp[ii] = abs(X_analytic).astype('float32')

            # Scales signals back to Volt
            X /= 1e6


            band_param_0V = VectorData(name='filter_param_0',
                              description='frequencies for bandpass filters',
                              data=band_param_0)
            band_param_1V = VectorData(name='filter_param_1',
                              description='frequencies for bandpass filters',
                              data=band_param_1)
            bandsTable = DynamicTable(name='bands',
                                      description='Series of filters used for Hilbert transform.',
                                      columns=[band_param_0V,band_param_1V],
                                      colnames=['filter_param_0','filter_param_1'])

            # data: (ndarray) dims: num_times * num_channels * num_bands
            Xp = np.swapaxes(Xp,0,2)
            decs = DecompositionSeries(name='Bandpower_'+filter,
                                        data=Xp,
                                        description='Band power estimated with Hilbert transform.',
                                        metric='power',
                                        unit='V**2/Hz',
                                        bands=bandsTable,
                                        rate=rate,
                                        source_timeseries=lfp_ts)
            ecephys_module.add_data_interface(decs)
        io.write(nwb)

        print('done', flush=True)
