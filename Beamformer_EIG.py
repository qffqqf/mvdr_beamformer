# -*- coding: utf-8 -*-
import numpy as np
import utils

class beamforimer_EIG:
    
    def __init__(self,
                 number_of_mic,
                 mic_distance,
                 sound_speed=343,
                 sampling_rate=16000,
                 fft_length=1024,
                 fft_shift=512):

        frequency_grid = np.linspace(0, sampling_rate, fft_length)
        frequency_grid = frequency_grid[0:int(fft_length / 2) + 1]
        
        self.frequency_grid = frequency_grid
        self.distance_vector = np.array([mic_distance]) * np.ones([number_of_mic,1]) # sensor distance
        # self.mic_vector = np.arange(0,number_of_mic,1).reshape([-1,1]) # N array from 0 to N-1
        self.number_of_mic=number_of_mic
        self.sound_speed=sound_speed
        self.sampling_frequency=sampling_rate
        self.fft_length=fft_length
        self.fft_shift=fft_shift
        
    def get_steering_vector(self, look_direction):
        steering_vector = np.ones((len(self.frequency_grid), self.number_of_mic), dtype=np.complex64)
        look_direction = look_direction * (-1)
        for f, frequency in enumerate(self.frequency_grid):
            for m, mic_pos in enumerate(self.distance_vector):
                steering_vector[f, m] = np.complex(np.exp(1j * 2 * np.pi * np.sin(np.deg2rad(look_direction)) * m * mic_pos / self.sound_speed * frequency))
        steering_vector = np.conjugate(steering_vector).T
        normalize_steering_vector = self.normalize(steering_vector)
        return normalize_steering_vector

    def get_single_steering_vector(self, look_direction, frequency):
        steering_vector = np.ones((1, self.number_of_mic), dtype=np.complex64)
        look_direction = look_direction * (-1)
        for m, mic_pos in enumerate(self.distance_vector):
            steering_vector[0, m] = np.complex(np.exp(1j * 2 * np.pi * np.sin(np.deg2rad(look_direction)) * m * mic_pos / self.sound_speed * frequency))
        steering_vector = np.conjugate(steering_vector).T
        weight = np.matmul(np.conjugate(steering_vector[:, 0]).T, steering_vector[:, 0])
        normalize_steering_vector = steering_vector /weight
        return normalize_steering_vector
    
    def normalize(self, steering_vector):
        for ii in range(0, self.frequency_grid.size):          
            weight = np.matmul(np.conjugate(steering_vector[:, ii]).T, steering_vector[:, ii])
            steering_vector[:, ii] = (steering_vector[:, ii] / weight) 
        return steering_vector
    
    def apply_beamformer(self, beamformer, complex_spectrum):
        number_of_channels, number_of_frames, number_of_bins = np.shape(complex_spectrum)        
        enhanced_spectrum = np.zeros((number_of_frames, number_of_bins), dtype=np.complex64)
        for f in range(0, number_of_bins):
            enhanced_spectrum[:, f] = np.matmul(np.conjugate(beamformer[:, f]).T, complex_spectrum[:, :, f])
        return utils.spec2wav(enhanced_spectrum, self.sampling_frequency, self.fft_length, self.fft_length, self.fft_shift)       