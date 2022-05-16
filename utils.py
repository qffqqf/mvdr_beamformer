# -*- coding: utf-8 -*-
import numpy as np
import h5py
from scipy.fftpack import fft, ifft
from scipy import signal as sg


def get_data_from_h5(filename, skip_last_channel=False):
    if skip_last_channel:
        if_skip = 1
    else:
        if_skip = 0
    f = h5py.File(filename, 'r')
    dset = f['time_data']
    source_signal = np.empty([dset.shape[0],dset.shape[1]-if_skip], dtype=float)
    for ind in np.arange(dset.shape[1]-if_skip):
        source_signal[:,ind] = dset[:,ind]
    return source_signal

def get_spectrogram(wav_data, frame, shift, fftl):
    len_sample, len_channel_vec = np.shape(wav_data)            
    dump_wav = wav_data.T
    dump_wav = dump_wav / np.max(np.abs(dump_wav)) * 0.7
    # window = sg.windows.hann(fftl + 1, sym=False)[: - 1]
    # multi_window = np.matlib.repmat(window, len_channel_vec, 1)
    st = 0
    ed = frame
    number_of_frame = int((len_sample - frame) / shift)
    spectrums = np.zeros((len_channel_vec, number_of_frame, int(fftl / 2) + 1), dtype=np.complex64)
    for ii in range(0, number_of_frame):
        # windowed_signal = dump_wav[:, st:ed] * multi_window
        windowed_signal = dump_wav[:, st:ed]
        multi_signal_spectrum = fft(windowed_signal, n=fftl, axis=1)[:, 0:int(fftl / 2) + 1] # channel * number_of_bin
        spectrums[:, ii, :] = multi_signal_spectrum
        st = st + shift
        ed = ed + shift
    return spectrums

def spec2wav(spectrogram, sampling_frequency, fftl, frame_len, shift_len):
    n_of_frame, fft_half = np.shape(spectrogram)
    # shift_len = fft_half
    # hanning_inv = 1 - sg.windows.hann(fftl + 1, sym=False)[: - 1]
    hanning = sg.windows.hann(fftl + 1, sym=False)[: - 1]
    cut_data = np.zeros(fftl, dtype=np.complex64)
    result = np.zeros(sampling_frequency * 60 * 5, dtype=np.float32)
    start_point = 0
    end_point = start_point + frame_len
    for ii in range(0, n_of_frame):
        half_spec = spectrogram[ii, :]
        cut_data[0:int(fftl / 2) + 1] = half_spec.T
        cut_data[int(fftl / 2) + 1:] =  np.flip(np.conjugate(half_spec[1:int(fftl / 2)]), axis=0)
        cut_data2 = np.real(ifft(cut_data, n=fftl))
        result[start_point:end_point] = result[start_point:end_point] + np.real(cut_data2)
        # result[start_point:end_point] = result[start_point:end_point] + np.real(cut_data2 * hanning.T)
        start_point = start_point + shift_len
        end_point = end_point + shift_len
    return result[0:end_point - shift_len]

def getSES(time_signal, sampling_rate, scaling_factor=2):
    sig_lenth = len(time_signal)
    f = np.linspace(0, sampling_rate, sig_lenth)
    analytic_signal = sg.hilbert(time_signal)
    amplitude_envelope = np.abs(analytic_signal)
    ses = np.abs(fft(amplitude_envelope) / sig_lenth)
    ses = ses[:int(sig_lenth/scaling_factor)]
    freqs = f[:int(sig_lenth/scaling_factor)]
    return ses, freqs

def demod(multi_signal):
    sig_shape = multi_signal.shape
    demod_sig = np.empty([sig_shape[0],sig_shape[1]], dtype=float)
    for ind_chn in np.arange(sig_shape[1]):
        demod_sig[:,ind_chn] = np.abs(sg.hilbert(multi_signal[:,ind_chn]))
    return demod_sig