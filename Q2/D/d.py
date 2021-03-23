from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.lib.function_base import interp
import math
import matplotlib.markers as markers
import matplotlib.patches as pat

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

def euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def kalman_filter(mu, sigma, z, A, U, B, R, C, Q):
    mu_t_dash = A.dot(mu) + B.dot(U)
    sigma_t_dash = A.dot(sigma).dot(A.T) + R

    K_t = sigma_t_dash.dot(C.T).dot(np.linalg.pinv(C.dot(sigma_t_dash).dot(C.T) + Q))
    mu_t = mu_t_dash + K_t.dot(z - C.dot(mu_t_dash))
    sigma_t = (np.eye(4, 4) - K_t.dot(C)).dot(sigma_t_dash)
    return mu_t, sigma_t

# U = np.zeros((2, 1))
def U(Omega, t):
    return np.array([np.sin(Omega*t * np.pi / 180.), np.cos(Omega*t * np.pi / 180.)]).reshape((2, 1))

C = np.eye(2, 4)
R = np.zeros((4, 4))
R[0, 0] = 1
R[1, 1] = 1
R[2, 2] = 0.0001
R[3, 3] = 0.0001
Q = 10 * np.eye(2, 2)
Omega = 30

init_location = 10 * np.ones((2, 1))
init_velocity = np.ones((2, 1))

init_state = np.vstack((init_location, init_velocity))
my_airplane = Airplane(init_state)
my_airplane.observed(observation_model(my_airplane.s, C, np.random.multivariate_normal(np.zeros(2,), Q).reshape(2, 1)))

T = 20
delta_t = 1

for t in range(T):
    S = motion_model(my_airplane.s, A(delta_t), B(delta_t), U(Omega, t), np.random.multivariate_normal(np.zeros(4,), R).reshape(4, 1))
    my_airplane.transition(S)
    my_airplane.observed(observation_model(my_airplane.s, C, np.random.multivariate_normal(np.zeros(2,), Q).reshape(2, 1)))

x_motion = [my_airplane.state_history[j][0, 0] for j in range(T)]
y_motion = [my_airplane.state_history[j][1, 0] for j in range(T)]
x_obs = [my_airplane.observation[j][0, 0] for j in range(T)]
y_obs = [my_airplane.observation[j][1, 0] for j in range(T)]
x_min = min(min(x_motion), min(x_obs))
x_max = max(max(x_motion), max(x_obs))
y_max = max(max(y_motion), max(y_obs))
y_min = min(min(y_motion), min(y_obs))

mu_0 = init_state
sigma_0 = 0.01 * np.eye(4, 4)

Bel = [(mu_0, sigma_0)]

for t in range(T):
    mu, sigma = kalman_filter(Bel[t][0], Bel[t][1], np.array([[x_obs[t]], [y_obs[t]]]), A(delta_t), U(Omega, t), B(delta_t), R, C, Q)
    Bel.append((mu, sigma))

predicted_state = [Bel[i][0] for i in range(len(Bel))]
predicted_state = np.array(predicted_state)

# Bar graph for euclidean error
fig2 = plt.figure(figsize = (10, 10))
euclidean_error = [euclidean((x_motion[i], y_motion[i]), (predicted_state[i, 0, 0], predicted_state[i, 1, 0])) for i in range(len(x_motion))]
time_stamps = [i*delta_t for i in range(len(x_motion))]
plt.bar(time_stamps, euclidean_error)
plt.xticks(np.arange(0, T, delta_t))
plt.xlabel("Time stamps")
plt.ylabel("Euclidean error between true and estimated paths")
plt.title("Variation of euclidean error with respect to time")

plt.savefig('d.png', bbox_inches='tight')

# Animation code
fig, ax = plt.subplots(1, 1, figsize = (10, 10))
ax.set_xticks(np.arange(x_min-10, x_max+10, 10))
ax.set_yticks(np.arange(y_min-10, y_max+10, 10))
ax.set_xlim([x_min-10, x_max+10])
ax.set_ylim([y_min-10, y_max+10])

line1, = ax.plot(x_motion, y_motion, color='g')
line2, = ax.plot(x_obs, y_obs, color='r')
line3, = ax.plot(predicted_state[0, 0, 0], predicted_state[0, 1, 0], color='b')

scat1 = plt.scatter([x_motion[0]], [y_motion[0]], c='g', s=50, edgecolors='k')
scat2 = plt.scatter([x_obs[0]], [y_obs[0]], c='r', s=50, edgecolors='k')
scat3 = plt.scatter([predicted_state[0, 0, 0]], [predicted_state[0, 1, 0]], c='b', s=50, edgecolors='k')

line1.set_label('Actual motion')
line2.set_label('Observed motion')
line3.set_label('Predicted motion')
plt.grid()
plt.legend(loc="upper left")

def animate(i):
    line1.set_data(x_motion[:i+1], y_motion[:i+1])
    line2.set_data(x_obs[:i+1], y_obs[:i+1])
    line3.set_data(predicted_state[:i+1, 0, 0], predicted_state[:i+1, 1, 0])
    scat1.set_offsets(np.c_[[x_motion[i]], [y_motion[i]]])
    scat2.set_offsets(np.c_[[x_obs[i]], [y_obs[i]]])
    scat3.set_offsets(np.c_[[predicted_state[i, 0, 0]], [predicted_state[i, 1, 0]]])
    return line1, line2, line3, scat1, scat2, scat3,

ani = animation.FuncAnimation(fig, animate, T, interval=delta_t * 1000, blit=True)
ani.save('d.mp4')

plt.show()