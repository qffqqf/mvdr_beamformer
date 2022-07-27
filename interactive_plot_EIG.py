import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy import linalg

def f(LOOK_DIRECTION, SIG_FREQ, N_ele):
    SOUND_SPEED = 343
    WAVE_LENGTH = SOUND_SPEED / SIG_FREQ
    ANGLE = 30
    Ny = 2
    Nx = N_ele-2*Ny
    Lx = 0.14
    Ly = 0.1
    alpha_left = np.deg2rad(np.arange(-90, LOOK_DIRECTION-ANGLE/2, 0.5).reshape([1, -1]))
    alpha_right = np.deg2rad(np.arange(LOOK_DIRECTION+ANGLE/2, 90, 0.5).reshape([1, -1]))
    alpha_out = np.concatenate( (alpha_left,alpha_right), axis=1)
    alpha_in = np.deg2rad(np.arange(LOOK_DIRECTION-ANGLE/2, LOOK_DIRECTION+ANGLE/2, 0.5).reshape([1, -1]))
    theta = np.deg2rad(np.arange(-90, 90, 0.1).reshape([1, -1]))
    mic_x_1 = -np.ones([Ny,1])*Lx/2
    mic_x_2 = np.ones([Ny,1])*Lx/2
    mic_x_3 = np.transpose(np.array([np.linspace(-Lx/2,Lx/2,Nx+2)]))
    mic_x_3 = mic_x_3[1:Nx+1] + 0.01*Lx*(np.random.rand(Nx,1)-0.5)/Nx
    mic_y_1 = np.transpose(np.array([np.linspace(Ly/2,-Ly/2,Ny)]))
    mic_y_2 = np.transpose(np.array([np.linspace(Ly/2,-Ly/2,Ny)]))
    mic_y_3 = np.ones([Nx,1])*Ly/2
    mic_x = np.concatenate( (np.concatenate((mic_x_1, mic_x_2), axis=0),mic_x_3), axis=0)
    mic_y = np.concatenate( (np.concatenate((mic_y_1, mic_y_2), axis=0),mic_y_3), axis=0)
    a = np.mat(np.exp(1j * 2* np.pi * (np.sin(theta)*mic_x + np.cos(theta)*mic_y) / WAVE_LENGTH))
    a_out = np.mat(np.exp(1j * 2* np.pi * (np.sin(alpha_out)*mic_x + np.cos(alpha_out)*mic_y) / WAVE_LENGTH))
    a_in = np.mat(np.exp(1j * 2* np.pi * (np.sin(alpha_in)*mic_x + np.cos(alpha_in)*mic_y) / WAVE_LENGTH))
    A_in = a_in* np.conjugate(a_in.T)
    A_out = a_out* np.conjugate(a_out.T) 
    [eigval, eigvec] = linalg.eig(A_out, A_in, right=True)
    min_index = np.argmin(np.abs(eigval))
    beamformer = eigvec[:,min_index]
    B = beamformer.conj().T * a
    B = np.abs(B) / np.max(np.abs(B)) 
    y = []
    for ele in np.arange(B.shape[1]):
        y.append(B[0,ele])
    
    return 20 * np.log10(y)

x=np.arange(-90, 90, 0.1).reshape([-1])

# Define initial parameters
init_direction = 0
init_frequency = 1000
init_elements = 6

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
line, = plt.plot(x, f(init_direction, init_frequency, init_elements), lw=2)
ax.set_xlabel('Degree[°]')
ax.set_ylabel('SPL[dB]')
ax.set_ylim(-70,0)

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

# define the values to use for snapping
allowed_vals = [i for i in range(2,17)]
axelem = plt.axes([0.25, 0.025, 0.5, 0.04])
elem_slider = Slider(
    ax=axelem,
    label='number of elements',
    valmin=2,
    valmax=16,
    valinit=init_elements,
    valstep=allowed_vals,
)

# Make a vertically oriented slider to control the amplitude
axdirec = plt.axes([0.1, 0.25, 0.0225, 0.63])
direc_slider = Slider(
    ax=axdirec,
    label="Direction",
    valmin=-90,
    valmax=90,
    valinit=init_direction,
    orientation="vertical"
)


# The function to be called anytime a slider's value changes
def update(val):
    line.set_ydata(f(direc_slider.val, freq_slider.val, elem_slider.val))
    fig.canvas.draw_idle()


# register the update function with each slider
freq_slider.on_changed(update)
direc_slider.on_changed(update)
elem_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    freq_slider.reset()
    direc_slider.reset()
    elem_slider.reset()
button.on_clicked(reset)

plt.show()