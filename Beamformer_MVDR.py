# -*- coding: utf-8 -*-
import numpy as np
from math import dist
from scipy.fftpack import fft
import utils


class beamformer_MVDR:

    def __init__(self,
                 mic_position,
                 sampling_rate=51200,
                 fft_length=512,
                 fft_shift=256,
                 sound_speed=343):
        frequency_grid = np.linspace(0, sampling_rate, fft_length)
        frequency_grid = frequency_grid[0:int(fft_length / 2) + 1]
        number_of_mic = len(mic_position)
        
        self.frequency_grid = frequency_grid
        self.number_of_mic = number_of_mic
        self.mic_position = mic_position
        self.sampling_rate = sampling_rate
        self.fft_length = fft_length
        self.fft_shift = fft_shift
        self.sound_speed = sound_speed

    def get_single_steering_vector_near_field(self, look_position, frequency):
        if frequency != 0:
            wavelength = self.sound_speed / frequency
            r_0p = dist(self.mic_position[0], look_position)
            r_mp = []
            a = []
            for i in np.arange(self.number_of_mic):
                r_ip = dist(self.mic_position[i], look_position)
                delta_r_mp = r_0p - r_ip
                a_i = r_0p / r_ip * np.exp(-1 * 1j * 2 * np.pi * delta_r_mp / wavelength)
                r_mp.append(r_ip)
                a.append(a_i)
        else:
            r_0p = dist(self.mic_position[0], look_position)
            r_mp = []
            a = []
            for i in np.arange(self.number_of_mic):
                r_ip = dist(self.mic_position[i], look_position)
                a_i = r_0p / r_ip
                r_mp.append(r_ip)
                a.append(a_i)
        a = np.array(a).reshape([1,self.number_of_mic])
        # weight = np.matmul(a, np.conjugate(a).T)
        # a_normalized = a / weight
        return a

    def get_steering_vector_near_field(self, look_position):
        steering_vector = np.empty((len(self.frequency_grid), self.number_of_mic), dtype=np.complex64)
        for f, frequency in enumerate(self.frequency_grid):
            steering_vector[f,:] = self.get_single_steering_vector_near_field(look_position, frequency)
        return steering_vector
    
    def get_spatial_correlation_matrix(self, multi_signal, use_number_of_frames_init=10, use_number_of_frames_final=10, tolerance=1e-8):
        # init
        start_index = 0
        end_index = start_index + self.fft_length
        record_length, number_of_channels = np.shape(multi_signal)
        R_mean = np.zeros((self.number_of_mic, self.number_of_mic, len(self.frequency_grid)), dtype=np.complex64)
        used_number_of_frames = 0
        
        # forward
        for _ in range(0, use_number_of_frames_init):
            multi_signal_cut = multi_signal[start_index:end_index, :]
            complex_signal = fft(multi_signal_cut, n=self.fft_length, axis=0)
            for f in range(0, len(self.frequency_grid)):
                    R_mean[:, :, f] = R_mean[:, :, f] + \
                        np.outer(complex_signal[f, :], np.conjugate(complex_signal[f, :]).T)
            used_number_of_frames = used_number_of_frames + 1
            start_index = start_index + self.fft_shift
            end_index = end_index + self.fft_shift
            if record_length <= start_index or record_length <= end_index:
                used_number_of_frames = used_number_of_frames - 1
                break
        
        # backward
        end_index = record_length
        start_index = end_index - self.fft_length
        for _ in range(0, use_number_of_frames_final):
            multi_signal_cut = multi_signal[start_index:end_index, :]
            complex_signal = fft(multi_signal_cut, n=self.fft_length, axis=0)
            for f in range(0, len(self.frequency_grid)):
                R_mean[:, :, f] = R_mean[:, :, f] + \
                    np.outer(complex_signal[f, :], np.conjugate(complex_signal[f, :]).T)
            used_number_of_frames = used_number_of_frames + 1
            start_index = start_index - self.fft_shift
            end_index = end_index - self.fft_shift            
            if  start_index < 1 or end_index < 1:
                used_number_of_frames = used_number_of_frames - 1
                break

        R_mean = R_mean / used_number_of_frames
        R_mean[abs(R_mean) < tolerance] = 0.0

        return R_mean
    
    def get_mvdr_beamformer(self, steering_vector, R, isDiagonalLoading):
        beamformer = np.empty((self.number_of_mic, len(self.frequency_grid)), dtype=np.complex64)
        for f in range(0, len(self.frequency_grid)):
            R_cut = np.reshape(R[:, :, f], [self.number_of_mic, self.number_of_mic])
            R_cut = np.mat(R_cut)
            #diagonal loading for singular matrix
            if isDiagonalLoading:
                R_cut = R_cut + np.eye(self.number_of_mic, dtype=int)
            steering_vector_cut = steering_vector[f,:]
            steering_vector_cut = np.mat(steering_vector_cut).T
            beamformer[:, f] = np.asarray(R_cut.I * steering_vector_cut / (steering_vector_cut.H * R_cut.I * steering_vector_cut)).reshape(self.number_of_mic,)
        return beamformer
    
    def apply_beamformer(self, beamformer, complex_spectrum):
        number_of_channels, number_of_frames, number_of_bins = np.shape(complex_spectrum)        
        enhanced_spectrum = np.zeros((number_of_frames, number_of_bins), dtype=np.complex64)
        for f in range(0, number_of_bins):
            enhanced_spectrum[:, f] = np.matmul(np.conjugate(beamformer[:, f]).T, complex_spectrum[:, :, f])
        return enhanced_spectrum
    
    def get_filter(self, rect_grid, beamformer, frequency):
        steering_vector_unit = np.empty([rect_grid.gpos.shape[1], self.number_of_mic], dtype=np.complex64)
        f = (np.abs(self.frequency_grid - frequency)).argmin()
        print(f'Calculating result for {self.frequency_grid[f]}Hz')
        for ind_grid in np.arange(rect_grid.gpos.shape[1]):
            steering_vector_column = []
            steering_vector_column = self.get_single_steering_vector_near_field(rect_grid.gpos[:,ind_grid], frequency)
            steering_vector_unit[ind_grid, :] = steering_vector_column
        w = np.mat(beamformer[:, f].reshape([self.number_of_mic, 1]))
        B = w.H * steering_vector_unit.T
        B = np.abs(B) / np.max(np.abs(B)) # normalization
        return B

    def get_noise_array_response(self, noise_pos, noise_spec):
        noise_array_response = np.empty([len(self.frequency_grid), noise_spec.shape[0], self.number_of_mic], dtype=np.complex64)
        for ind_f in np.arange(len(self.frequency_grid)):
            steering_vector_column = []
            steering_vector_column = self.get_single_steering_vector_near_field(noise_pos, self.frequency_grid[ind_f])
            noise_freq = np.mat(noise_spec[:,ind_f]).reshape([noise_spec.shape[0], 1])
            noise_array_response[ind_f, :, :] = noise_freq * steering_vector_column
        return noise_array_response

    def get_augmented_noise(self, noise_pos, multi_signal, noise_ch_ind):
        noise_channel = np.mat(multi_signal[:,int(noise_ch_ind)].reshape([multi_signal[:,int(noise_ch_ind)].size, 1]))
        noise_spectrum = utils.get_spectrogram(noise_channel, self.fft_length, self.fft_shift, self.fft_length)[0,:,:]
        noise_array_response = self.get_noise_array_response(noise_pos, noise_spectrum)
        noise_audio = utils.spec2wav(noise_array_response[:,:,0].T, self.sampling_rate, self.fft_length, self.fft_length, self.fft_shift)
        noise_signal = np.empty([len(noise_audio), self.number_of_mic], dtype=np.float32)
        for ind_ch in np.arange(self.number_of_mic):
            noise_signal[:,ind_ch] = utils.spec2wav(noise_array_response[:,:,ind_ch].T, self.sampling_rate, self.fft_length, self.fft_length, self.fft_shift)
        return noise_signal


class beamformer_MVDR_FarField:
    
    def __init__(self,
                 mic_angle_vector,
                 mic_diameter,
                 sampling_frequency=16000,
                 fft_length=512,
                 fft_shift=256,
                 sound_speed=343):   

        frequency_grid = np.linspace(0, sampling_frequency, fft_length)
        frequency_grid = frequency_grid[0:int(fft_length / 2) + 1]
        number_of_mic = len(mic_angle_vector)
        
        self.frequency_grid = frequency_grid
        self.number_of_mic = number_of_mic
        self.mic_angle_vector=mic_angle_vector
        self.mic_diameter=mic_diameter
        self.sampling_frequency=sampling_frequency
        self.fft_length=fft_length
        self.fft_shift=fft_shift
        self.sound_speed=sound_speed
    
    def get_single_steering_vector(self, look_direction, frequency):
        steering_vector = np.empty([1, self.number_of_mic], dtype=np.complex64)
        look_direction = look_direction * (-1)
        for m, mic_angle in enumerate(self.mic_angle_vector):
            steering_vector[0,m] = np.complex(np.exp(( - 1j) * ((2 * np.pi * frequency) / self.sound_speed) \
                                               * (self.mic_diameter / 2) \
                                               * np.cos(np.deg2rad(look_direction) - np.deg2rad(mic_angle))))
        steering_vector = np.conjugate(steering_vector).T
        # normalize_single_steering_vector = self.normalize(steering_vector)
        return steering_vector

    def get_steering_vector(self, look_direction):
        # frequency_vector = np.linspace(0, self.sampling_frequency, self.fft_length)
        steering_vector = np.ones((len(self.frequency_grid), self.number_of_mic), dtype=np.complex64)
        look_direction = look_direction * (-1)
        for f, frequency in enumerate(self.frequency_grid):
            for m, mic_angle in enumerate(self.mic_angle_vector):
                steering_vector[f, m] = np.complex(np.exp(( - 1j) * ((2 * np.pi * frequency) / self.sound_speed) \
                               * (self.mic_diameter / 2) \
                               * np.cos(np.deg2rad(look_direction) - np.deg2rad(mic_angle))))
        steering_vector = np.conjugate(steering_vector).T
        # normalize_steering_vector = self.normalize(steering_vector)
        # return normalize_steering_vector[0:np.int(self.fft_length / 2) + 1, :]
        return steering_vector  
    
    def normalize(self, steering_vector):
        for ii in range(0, self.fft_length):            
            weight = np.matmul(np.conjugate(steering_vector[:, ii]).T, steering_vector[:, ii])
            steering_vector[:, ii] = (steering_vector[:, ii] / weight) 
        return steering_vector        
    
    def get_spatial_correlation_matrix(self, multi_signal, use_number_of_frames_init=10, use_number_of_frames_final=10):
        # init
        # number_of_mic = len(self.mic_angle_vector)
        # frequency_grid = np.linspace(0, self.sampling_frequency, self.fft_length)
        # frequency_grid = frequency_grid[0:np.int(self.fft_length / 2) + 1]
        start_index = 0
        end_index = start_index + self.fft_length
        speech_length, number_of_channels = np.shape(multi_signal)
        R_mean = np.zeros((self.number_of_mic, self.number_of_mic, len(self.frequency_grid)), dtype=np.complex64)
        used_number_of_frames = 0
        
        # forward
        for _ in range(0, use_number_of_frames_init):
            multi_signal_cut = multi_signal[start_index:end_index, :]
            complex_signal = fft(multi_signal_cut, n=self.fft_length, axis=0)
            for f in range(0, len(self.frequency_grid)):
                    R_mean[:, :, f] = R_mean[:, :, f] + \
                        np.multiply.outer(complex_signal[f, :], np.conj(complex_signal[f, :]).T)
            used_number_of_frames = used_number_of_frames + 1
            start_index = start_index + self.fft_shift
            end_index = end_index + self.fft_shift
            if speech_length <= start_index or speech_length <= end_index:
                used_number_of_frames = used_number_of_frames - 1
                break            
        
        # backward
        end_index = speech_length
        start_index = end_index - self.fft_length
        for _ in range(0, use_number_of_frames_final):
            multi_signal_cut = multi_signal[start_index:end_index, :]
            complex_signal = fft(multi_signal_cut, n=self.fft_length, axis=0)
            for f in range(0, len(self.frequency_grid)):
                R_mean[:, :, f] = R_mean[:, :, f] + \
                    np.multiply.outer(complex_signal[f, :], np.conj(complex_signal[f, :]).T)
            used_number_of_frames = used_number_of_frames + 1
            start_index = start_index - self.fft_shift
            end_index = end_index - self.fft_shift            
            if  start_index < 1 or end_index < 1:
                used_number_of_frames = used_number_of_frames - 1
                break                    

        return R_mean / used_number_of_frames  
    
    def get_mvdr_beamformer(self, steering_vector, R):
        # number_of_mic = len(self.mic_angle_vector)
        # frequency_grid = np.linspace(0, self.sampling_frequency, self.fft_length)
        # frequency_grid = frequency_grid[0:np.int(self.fft_length / 2) + 1]        
        beamformer = np.ones((self.number_of_mic, len(self.frequency_grid)), dtype=np.complex64)
        for f in range(0, len(self.frequency_grid)):
            R_cut = np.reshape(R[:, :, f], [self.number_of_mic, self.number_of_mic])
            inv_R = np.linalg.pinv(R_cut)
            a = np.matmul(np.conjugate(steering_vector[:, f]).T, inv_R)
            b = np.matmul(a, steering_vector[:, f])
            b = np.reshape(b, [1, 1])
            beamformer[:, f] = np.matmul(inv_R, steering_vector[:, f]) / b # number_of_mic *1   = number_of_mic *1 vector/scalar        
        return beamformer
    
    def apply_beamformer(self, beamformer, complex_spectrum):
        number_of_channels, number_of_frames, number_of_bins = np.shape(complex_spectrum)        
        enhanced_spectrum = np.zeros((number_of_frames, number_of_bins), dtype=np.complex64)
        for f in range(0, number_of_bins):
            enhanced_spectrum[:, f] = np.matmul(np.conjugate(beamformer[:, f]).T, complex_spectrum[:, :, f])
        return utils.spec2wav(enhanced_spectrum, self.sampling_frequency, self.fft_length, self.fft_length, self.fft_shift)
    
    def get_filter(self, search_angle_vec, beamformer, frequency):
        steering_vector_unit = np.empty([search_angle_vec.shape[1], self.number_of_mic], dtype=np.complex64)
        f = (np.abs(self.frequency_grid - frequency)).argmin()
        print(f'Calculating result for {self.frequency_grid[f]}Hz')
        for ind_ang in np.arange(search_angle_vec.shape[1]):
            steering_vector_column = []
            steering_vector_column = self.get_single_steering_vector(search_angle_vec[0,ind_ang], frequency)
            steering_vector_unit[ind_ang, :] = steering_vector_column[:,0]
        w = np.mat(beamformer[:, f].reshape([self.number_of_mic, 1]))
        B = w.H * steering_vector_unit.T
        B = np.abs(B) / np.max(np.abs(B)) # normalization
        return B
    
    def get_single_filter(self, search_angle_vec, beamformer, frequency):
        steering_vector_unit = np.empty([search_angle_vec.shape[1], self.number_of_mic], dtype=np.complex64)
        for ind_ang in np.arange(search_angle_vec.shape[1]):
            steering_vector_column = []
            steering_vector_column = self.get_single_steering_vector(search_angle_vec[0,ind_ang], frequency)
            steering_vector_unit[ind_ang, :] = steering_vector_column[:,0]
        w = np.mat(beamformer.reshape([self.number_of_mic, 1]))
        B = w.H * steering_vector_unit.T
        B = np.abs(B) / np.max(np.abs(B)) # normalization
        return B