import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.lib.function_base import interp

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


U = np.zeros((2, 1))
C = np.eye(2, 4)
R = np.zeros((4, 4))
R[0, 0] = 1
R[1, 1] = 1
R[2, 2] = 0.0001
R[3, 3] = 0.0001
Q = 10 * np.eye(2, 2)

movement_dimension = 50
movement_scale = 5

init_location = movement_dimension * np.random.rand(2, 1)
init_velocity = movement_scale * np.random.rand(2, 1)

init_state = np.vstack((init_location, init_velocity))
my_airplane = Airplane(init_state)
my_airplane.observed(observation_model(my_airplane.s, C, np.random.multivariate_normal(np.zeros(2,), Q).reshape(2, 1)))

T = 50
delta_t = 0.5

for t in range(T):
    S = motion_model(my_airplane.s, A(delta_t), B(delta_t), U, np.random.multivariate_normal(np.zeros(4,), R).reshape(4, 1))
    my_airplane.transition(S)
    my_airplane.observed(observation_model(my_airplane.s, C, np.random.multivariate_normal(np.zeros(2,), Q).reshape(2, 1)))

x_motion = [my_airplane.state_history[j][0, 0] for j in range(T)]
y_motion = [my_airplane.state_history[j][1, 0] for j in range(T)]
x_obs = [my_airplane.observation[j][0, 0] for j in range(T)]
y_obs = [my_airplane.observation[j][1, 0] for j in range(T)]


fig, ax = plt.subplots(1, 1, figsize = (10, 10))
ax.set_xlim([0, movement_dimension + T*movement_scale/3])
ax.set_ylim([0, movement_dimension + T*movement_scale/3])
line1, = ax.plot(x_motion, y_motion, color='g')
line2, = ax.plot(x_obs, y_obs, color='r')
scat1 = plt.scatter([x_motion[0]], [y_motion[0]], c='g', s=50, edgecolors='k')
scat2 = plt.scatter([x_obs[0]], [y_obs[0]], c='r', s=50, edgecolors='k')

def animate(i):
    line1.set_data(x_motion[:i], y_motion[:i])
    line2.set_data(x_obs[:i], y_obs[:i])
    if i == 0:
        scat1.set_offsets(np.c_[[x_motion[0]], [y_motion[0]]])
        scat2.set_offsets(np.c_[[x_obs[0]], [y_obs[0]]])
    else:
        scat1.set_offsets(np.c_[[x_motion[i-1]], [y_motion[i-1]]])
        scat2.set_offsets(np.c_[[x_obs[i-1]], [y_obs[i-1]]])
    return line1, line2, scat1, scat2,

ani = animation.FuncAnimation(fig, animate, T, interval=delta_t * 1000, blit=True)
ani.save('a.mp4')

plt.show()