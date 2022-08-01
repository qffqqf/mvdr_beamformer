import numpy as np
from scipy import linalg

TARGET = np.arange([0,1]), SIG_FREQ, N_ele
SOUND_SPEED = 343
WAVE_LENGTH = SOUND_SPEED / SIG_FREQ
ANGLE = 90
Ny = 2
Nx = N_ele-2*Ny
Lx = 0.14
Ly = 0.05
alpha_left = np.deg2rad(np.arange(-90, LOOK_DIRECTION-ANGLE/4, 0.5).reshape([1, -1]))
alpha_right = np.deg2rad(np.arange(LOOK_DIRECTION+ANGLE/4, 90, 0.5).reshape([1, -1]))
alpha_out = np.concatenate( (alpha_left,alpha_right), axis=1)
alpha_in = np.deg2rad(np.arange(LOOK_DIRECTION-ANGLE/2, LOOK_DIRECTION+ANGLE/2, 0.5).reshape([1, -1]))
theta = np.deg2rad(np.arange(-90, 90, 0.1).reshape([1, -1]))
mic_x_1 = -np.ones([Ny,1])*Lx/2
mic_x_2 = np.ones([Ny,1])*Lx/2
mic_x_3 = np.transpose(np.array([np.linspace(-Lx/2,Lx/2,Nx+2)]))
mic_x_3 = mic_x_3[1:Nx+1] + 0.01*Lx*(np.random.rand(Nx,1)-0.5)/Nx
mic_y_1 = np.transpose(np.array([np.linspace(Ly/2,-Ly/2,Ny)])) - 0.2*Ly*np.random.rand(Ny,1)/Ny
mic_y_2 = np.transpose(np.array([np.linspace(Ly/2,-Ly/2,Ny)])) - 0.2*Ly*np.random.rand(Ny,1)/Ny
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








'''
from crypt import methods
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, antialiased=False, linewidth=0)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
'''