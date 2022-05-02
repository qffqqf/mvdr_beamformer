from Beamformer_MVDR import beamformer_MVDR
import utils
from acoular import MicGeom
import numpy as np
from scipy import signal, stats
from acoular import RectGrid
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import os
from pylab import axis, imshow, colorbar, title, savefig, show


def getIndicator(look_pos):
    steering_vector = mvdr_beamformer.get_steering_vector_near_field(look_pos)
    beamformer = mvdr_beamformer.get_mvdr_beamformer(steering_vector, spatial_correlation_matrix, isDiagonalLoading)
    enhanced_spectrum = mvdr_beamformer.apply_beamformer(beamformer, complex_spectrum)
    enhanced_audio = utils.spec2wav(enhanced_spectrum, SAMPLING_RATE, FFT_LENGTH, FFT_LENGTH, FFT_SHIFT)
    data = enhanced_audio / np.max(np.abs(enhanced_audio)) * 0.7
    ses, freqs = utils.getSES(data, SAMPLING_RATE, 50)
    mod_freq = 850
    ind_mod = (np.abs(freqs - mod_freq)).argmin()
    # noise_freq = 400
    # ind_mod_noise = (np.abs(freqs - noise_freq)).argmin()
    thres1 = signal.medfilt(ses, 255)
    thres2 = stats.median_abs_deviation(ses)
    thres3 = thres1 + 6 * thres2
    indi_temp = round(ses[ind_mod] / thres3.mean(), 2)
    # result = [look_pos[0], look_pos[1], indi_temp]
    return indi_temp

# Parameters

INCREMENT = 0.2
isDiagonalLoading = False

test_name = "scan"
MIX_FILENAME = 'mix_mod_2_large.h5'
SIG_FILENAME = 'sig_mod_64.h5'
IAN_FILENAME = 'noise_mod_64.h5'
dir_path = './data'
MIX_FILENAME = os.path.join(dir_path, MIX_FILENAME)
SIG_FILENAME = os.path.join(dir_path, SIG_FILENAME)
IAN_FILENAME = os.path.join(dir_path, IAN_FILENAME)

OUTPUT_FOLDER = f'result_increment_{INCREMENT}'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

RESULT_NAME = f'result_increment_source_{INCREMENT}.npy'
FIGURE_NAME = f'figure_increment_source_{INCREMENT}.epi'
RESULT_NAME = os.path.join(OUTPUT_FOLDER, RESULT_NAME)
FIGURE_NAME = os.path.join(OUTPUT_FOLDER, FIGURE_NAME)

mg = MicGeom(from_file='./array_geom/array_2_large.xml')
number_of_mic = mg.mpos.shape[1]
MIC_POS = []
for i in np.arange(number_of_mic):
    MIC_POS.append(mg.mpos[:,int(i)])

LOOK_POS = [0.8,2.3,0.5]
NOISE_POS = [3.2,2.2,0.5]
# NOISE_CH = 51
# SIG_CH = 19
NOISE_CH = 1
SIG_CH = 0
SAMPLING_RATE = 51200
FFT_LENGTH = 8192
FFT_SHIFT = 2048
SOUND_SPEED = 343

mvdr_beamformer = beamformer_MVDR(MIC_POS, sampling_rate=SAMPLING_RATE, fft_length=FFT_LENGTH, fft_shift=FFT_SHIFT, sound_speed=SOUND_SPEED)
### noise spectrogram multiplied with steering vector pointing to its position
multi_signal = utils.get_data_from_h5(MIX_FILENAME, skip_last_channel=False)
noise_signal = mvdr_beamformer.get_augmented_noise(NOISE_POS, multi_signal, NOISE_CH)
spatial_correlation_matrix = mvdr_beamformer.get_spatial_correlation_matrix(noise_signal) # noise covariance matrix
steering_vector = mvdr_beamformer.get_steering_vector_near_field(LOOK_POS)
beamformer = mvdr_beamformer.get_mvdr_beamformer(steering_vector, spatial_correlation_matrix, isDiagonalLoading)
complex_spectrum = utils.get_spectrogram(multi_signal, FFT_LENGTH, FFT_SHIFT, FFT_LENGTH)
enhanced_spectrum = mvdr_beamformer.apply_beamformer(beamformer, complex_spectrum)

"""
define grid
"""
rg = RectGrid( x_min=-0.2, x_max=4.2,
                        y_min=-0.2, y_max=4.2,
                        z=0.5, increment=INCREMENT )

grid_points = int(np.sqrt(rg.gpos.shape[1]))

myList = [rg.gpos[:,ind] for ind in np.arange(rg.gpos.shape[1])]

num_cores = multiprocessing.cpu_count()
inputs = tqdm(myList)

# source mismatch
processed_list = Parallel(n_jobs=num_cores)(delayed(getIndicator)(i) for i in inputs)

results = np.array(processed_list)

with open(RESULT_NAME, 'wb') as f:
    np.save(f, results)

"""
plot beamformer
"""

Z_g = results.reshape(grid_points,grid_points)
# imshow( Z_g.T, origin='lower', extent=rg.extend(), interpolation='bicubic')
imshow( Z_g.T, origin='lower', extent=rg.extend())
colorbar()
axis('equal')
title('source location mismatch')
savefig(FIGURE_NAME, format='eps')
show()