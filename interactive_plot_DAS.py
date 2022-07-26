import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from Beamformer_DAS import beamforimer_DAS
import numpy as np

def f(LOOK_DIRECTION, SIG_FREQ):
    SAMPLING_RATE = 16000
    FFT_LENGTH = 2048
    FFT_SHIFT = 1024
    SOUND_SPEED = 343
    N = 4
    ELEMENT_DISTANCE = 0.15/(N-1)
    WAVE_LENGTH = SOUND_SPEED / SIG_FREQ
    theta = np.deg2rad(np.arange(-90, 90, 0.1).reshape([1, -1])) # every angle for filter curve
    d = np.array([ELEMENT_DISTANCE]) * np.ones([N,1]) # sensor distance
    n = np.arange(0,N,1).reshape([-1,1]) # N array from 0 to N-1
    a = np.mat(np.exp(1j * 2* np.pi * np.sin(theta) * n * d / WAVE_LENGTH))
    ds_beamformer = beamforimer_DAS(number_of_mic=N, mic_distance=ELEMENT_DISTANCE, sound_speed=SOUND_SPEED, sampling_rate=SAMPLING_RATE, fft_length=FFT_LENGTH, fft_shift=FFT_SHIFT)
    beamformer = ds_beamformer.get_single_steering_vector(LOOK_DIRECTION, SIG_FREQ)
    B = beamformer.T * a
    B = np.abs(B) / np.max(np.abs(B)) # normalization
    y = []
    for ele in np.arange(B.shape[1]):
        y.append(B[0,ele])
    
    return 20 * np.log10(y)

x=np.arange(-90, 90, 0.1).reshape([-1])

# Define initial parameters
init_direction = 0
init_frequency = 4000

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
line, = plt.plot(x, f(init_direction, init_frequency), lw=2)
ax.set_xlabel('Degree[°]')

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label='Frequency [Hz]',
    valmin=0.1,
    valmax=8000,
    valinit=init_frequency,
)

# Make a vertically oriented slider to control the amplitude
axamp = plt.axes([0.1, 0.25, 0.0225, 0.63])
direc_slider = Slider(
    ax=axamp,
    label="Direction",
    valmin=-90,
    valmax=90,
    valinit=init_direction,
    orientation="vertical"
)


# The function to be called anytime a slider's value changes
def update(val):
    line.set_ydata(f(direc_slider.val, freq_slider.val))
    fig.canvas.draw_idle()


# register the update function with each slider
freq_slider.on_changed(update)
direc_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    freq_slider.reset()
    direc_slider.reset()
button.on_clicked(reset)

plt.show()