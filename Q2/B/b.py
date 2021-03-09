import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.lib.function_base import interp
import math
import matplotlib.markers as markers

bf_sz = 30
sensor_cred = 0.7       # Less is better. Lesser this value, more sure we are about the sensor measurement.

class Airplane:
    def __init__(self, s):
        self.s = s
        self.state_history = [s]
        self.observation = []
    
    def transition(self, s):
        self.s = s
        self.state_history.append(s)

    def observed(self, s):
        self.observation.append(s)

def A(delta_t):
    I = np.eye(4, 4)
    I[0, 2] = delta_t
    I[1, 3] = delta_t
    return I

def B(delta_t):
    T = np.zeros((4, 2))
    T[0, 0] = delta_t*delta_t/2
    T[1, 1] = delta_t*delta_t/2
    T[2, 0] = 1
    T[3, 1] = 1
    return T


def motion_model(S, A, B, U, eps):
    return A.dot(S) + B.dot(U) + eps

def observation_model(S, C, delta):
    return C.dot(S) + delta

def chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

def sensor_model(z, x):
    return math.pow(sensor_cred, chebyshev(z, x))

def bayes_filter(Bel, TM, obs):
    Bel_dash = TM.dot(Bel)
    sensor_pref = np.zeros((bf_sz*bf_sz, 1))

    for k in range(bf_sz*bf_sz):
        i, j = k//bf_sz, k%bf_sz
        sensor_pref[k, 0] = sensor_model(obs, (i, j))
    
    Bel_dash = np.multiply(Bel_dash, sensor_pref)
    eta = np.sum(Bel_dash)
    Bel_new = Bel_dash/eta
    return Bel_new



U = np.zeros((2, 1))
C = np.eye(2, 4)
R = np.zeros((4, 4))
R[0, 0] = 1
R[1, 1] = 1
R[2, 2] = 0.0001
R[3, 3] = 0.0001
Q = 10 * np.eye(2, 2)

# movement_dimension = 50
# movement_scale = 5

# init_location = movement_dimension * np.random.rand(2, 1)
# init_velocity = movement_scale * np.random.rand(2, 1)
init_location = 10 * np.ones((2, 1))
init_velocity = np.ones((2, 1))

init_state = np.vstack((init_location, init_velocity))
my_airplane = Airplane(init_state)
my_airplane.observed(observation_model(my_airplane.s, C, np.random.multivariate_normal(np.zeros(2,), Q).reshape(2, 1)))


# def animate(i):
#     line1.set_data(x_motion[:i], y_motion[:i])
#     line2.set_data(x_obs[:i], y_obs[:i])
#     if i == 0:
#         scat1.set_offsets(np.c_[[x_motion[0]], [y_motion[0]]])
#         scat2.set_offsets(np.c_[[x_obs[0]], [y_obs[0]]])
#     else:
#         scat1.set_offsets(np.c_[[x_motion[i-1]], [y_motion[i-1]]])
#         scat2.set_offsets(np.c_[[x_obs[i-1]], [y_obs[i-1]]])
#     return line1, line2, scat1, scat2,

# ani = animation.FuncAnimation(fig, animate, T, interval=delta_t * 1000, blit=True)
# ani.save('a.mp4')

# plt.show()



T = 20
delta_t = 1

Bel = np.zeros((T, bf_sz*bf_sz, 1))
Bel[0, int(init_state[0, 0]*bf_sz + init_state[1, 0]), 0] = 1

TM = np.zeros((bf_sz*bf_sz, bf_sz*bf_sz))       # Transition Matrix

for i in range(bf_sz - 2):
    for j in range(bf_sz - 2):
        fr = bf_sz*i + j
        to1, to2, to3, to4, to5, to6, to7, to8, to9 = fr + bf_sz + 1, fr + 2*bf_sz + 1, fr + bf_sz + 2, fr + 1, fr + bf_sz, fr + 2*bf_sz + 2, fr + 2*bf_sz, fr, fr + 2
        TM[to1, fr] = 0.7
        TM[to2, fr], TM[to3, fr], TM[to4, fr], TM[to5, fr] = 0.05, 0.05, 0.05, 0.05
        TM[to6, fr], TM[to7, fr], TM[to8, fr], TM[to9, fr] = 0.025, 0.025, 0.025, 0.025

for i in range(bf_sz - 1):
    fr1 = bf_sz*i + bf_sz-2
    to1 = fr1 + bf_sz + 1
    TM[to1, fr1] = 1
    fr2 = bf_sz*(bf_sz-2) + i
    to2 = fr2 + bf_sz + 1
    TM[to2, fr2] = 1

for i in range(bf_sz):
    fr1 = bf_sz*i + bf_sz-1
    TM[fr1, fr1] = 1
    fr2 = bf_sz*(bf_sz-1) + i
    TM[fr2, fr2] = 1

for t in range(T):
    S = motion_model(my_airplane.s, A(delta_t), B(delta_t), U, np.random.multivariate_normal(np.zeros(4,), R).reshape(4, 1))
    my_airplane.transition(S)
    my_airplane.observed(observation_model(my_airplane.s, C, np.random.multivariate_normal(np.zeros(2,), Q).reshape(2, 1)))

x_motion = [my_airplane.state_history[j][0, 0] for j in range(T)]
y_motion = [my_airplane.state_history[j][1, 0] for j in range(T)]
x_obs = [my_airplane.observation[j][0, 0] for j in range(T)]
y_obs = [my_airplane.observation[j][1, 0] for j in range(T)]

for i in range(1, 20):
    Bel[i] = bayes_filter(Bel[i-1], TM, (x_obs[i], y_obs[i]))

predicted_path = []

for i in range(20):
    k = np.argmax(Bel[i])
    x, y = k//bf_sz, k%bf_sz
    predicted_path.append((x, y))

fig, ax = plt.subplots(1, 1, figsize = (10, 10))
ax.set_xticks(np.arange(0, bf_sz+1, 1))
ax.set_yticks(np.arange(0, bf_sz+1, 1))
ax.set_xlim([-0.5, bf_sz-0.5])
ax.set_ylim([-0.5, bf_sz-0.5])
line1, = ax.plot(x_motion, y_motion, color='g')
line2, = ax.plot(x_obs, y_obs, color='r')

P = np.arange(bf_sz)
all_points = np.dstack(np.meshgrid(P, P)).reshape(-1, 2)
marker = markers.MarkerStyle(marker='s')

scat3 = plt.scatter(all_points[:, 0], all_points[:, 1], c=Bel[0, :, 0]*1000, s=200, cmap='Greys', edgecolors='k', marker=marker)
scat1 = plt.scatter([x_motion[0]], [y_motion[0]], c='g', s=50, edgecolors='k')
scat2 = plt.scatter([x_obs[0]], [y_obs[0]], c='r', s=50, edgecolors='k')
scat4 = plt.scatter([10], [10], c='b', s=200, edgecolors='k', marker=marker)

line1.set_label('Actual motion')
line2.set_label('Observed motion')
plt.grid()
plt.legend(loc="upper left")

def animate(i):
    arr = []
    brr = []
    shade_map = []
    for a in range(bf_sz):
        for b in range(bf_sz):
            arr.append(a)
            brr.append(b)
            shade_map.append(Bel[i,a*bf_sz + b,0]*1000)

    scat3.set_offsets(np.c_[arr, brr])
    scat3.set_array(np.array(shade_map))
    line1.set_data(x_motion[:i+1], y_motion[:i+1])
    line2.set_data(x_obs[:i+1], y_obs[:i+1])
    scat1.set_offsets(np.c_[[x_motion[i]], [y_motion[i]]])
    scat2.set_offsets(np.c_[[x_obs[i]], [y_obs[i]]])
    scat4.set_offsets(np.c_[[predicted_path[i][0]], [predicted_path[i][1]]])
    return scat3, scat4, scat1, scat2, line1, line2,

ani = animation.FuncAnimation(fig, animate, T, interval=delta_t * 1000, blit=True)
ani.save('b.mp4')

plt.show()