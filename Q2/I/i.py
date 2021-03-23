
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.lib.function_base import interp
import math
import matplotlib.markers as markers
import matplotlib.patches as pat
import random


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




observation_table = []


C_a = np.eye(2, 4)
R_a = np.zeros((4, 4))
R_a[0, 0] = 1
R_a[1, 1] = 1
R_a[2, 2] = 0.0001
R_a[3, 3] = 0.0001
Q_a = 10 * np.eye(2, 2)
Omega_a = 30


C_b = np.eye(2, 4)
R_b = np.zeros((4, 4))
R_b[0, 0] = 1
R_b[1, 1] = 1
R_b[2, 2] = 0.0001
R_b[3, 3] = 0.0001
Q_b = 10 * np.eye(2, 2)
Omega_b = 30

init_location_a = 10 * np.ones((2, 1))
init_velocity_a = np.ones((2, 1))

init_location_b = 10 * np.ones((2,1))
init_velocity_b = np.ones((2, 1))

init_state_a = np.vstack((init_location_a, init_velocity_a))
my_airplane_a = Airplane(init_state_a)
observations_a = observation_model(my_airplane_a.s, C_a, np.random.multivariate_normal(np.zeros(2,), Q_a).reshape(2, 1))
my_airplane_a.observed(observations_a)

init_state_b = np.vstack((init_location_b, init_velocity_b))
my_airplane_b = Airplane(init_state_b)
observations_b = observation_model(my_airplane_b.s, C_b, np.random.multivariate_normal(np.zeros(2,), Q_b).reshape(2, 1))
my_airplane_b.observed(observations_b)

# print(observations_a)
# print([observations_a,observations_b])
arrp = [0,1]
random.shuffle(arrp)
arrobs = []
for i in range(2):
    if arrp[0]==0:
        arrobs.append(observations_a)
        arrobs.append(observations_b)
    else:
        arrobs.append(observations_b)
        arrobs.append(observations_a)
observation_table.append(arrobs)
# print(obs)
T = 20
delta_t = 1

for t in range(T):
    S_a = motion_model(my_airplane_a.s, A(delta_t), B(delta_t), U(Omega_a, t), np.random.multivariate_normal(np.zeros(4,), R_a).reshape(4, 1))
    my_airplane_a.transition(S_a)
    observations_a = observation_model(my_airplane_a.s, C_a, np.random.multivariate_normal(np.zeros(2,), Q_a).reshape(2, 1))
    my_airplane_a.observed(observations_a)
    S_b = motion_model(my_airplane_b.s, A(delta_t), B(delta_t), U(Omega_b, t), np.random.multivariate_normal(np.zeros(4,), R_b).reshape(4, 1))
    my_airplane_b.transition(S_b)
    observations_b = observation_model(my_airplane_b.s, C_b, np.random.multivariate_normal(np.zeros(2,), Q_b).reshape(2, 1))
    my_airplane_b.observed(observations_b)
    arrp = [0,1]
    random.shuffle(arrp)
    arrobs = []
    for i in range(2):
        if arrp[0]==0:
            arrobs.append(observations_a)
            arrobs.append(observations_b)
        else:
            arrobs.append(observations_b)
            arrobs.append(observations_a)
    observation_table.append(arrobs)

x_motion_a = [my_airplane_a.state_history[j][0, 0] for j in range(T)]
y_motion_a = [my_airplane_a.state_history[j][1, 0] for j in range(T)]
x_obs_a = [my_airplane_a.observation[j][0, 0] for j in range(T)]
y_obs_a = [my_airplane_a.observation[j][1, 0] for j in range(T)]
x_min_a = min(min(x_motion_a), min(x_obs_a))
x_max_a = max(max(x_motion_a), max(x_obs_a))
y_max_a = max(max(y_motion_a), max(y_obs_a))
y_min_a = min(min(y_motion_a), min(y_obs_a))


x_motion_b = [my_airplane_b.state_history[j][0, 0] for j in range(T)]
y_motion_b = [my_airplane_b.state_history[j][1, 0] for j in range(T)]
x_obs_b = [my_airplane_b.observation[j][0, 0] for j in range(T)]
y_obs_b = [my_airplane_b.observation[j][1, 0] for j in range(T)]
x_min_b = min(min(x_motion_b), min(x_obs_b))
x_max_b = max(max(x_motion_b), max(x_obs_b))
y_max_b = max(max(y_motion_b), max(y_obs_b))
y_min_b = min(min(y_motion_b), min(y_obs_b))

x_min_f = min(x_min_a, x_min_b)
x_max_f = max(x_max_a, x_max_b)
y_min_f = min(y_min_a, y_min_b)
y_max_f = max(y_max_a, y_max_b)

mu_0_a = init_state_a
sigma_0_a = 0.01 * np.eye(4, 4)

Bel_a = [(mu_0_a, sigma_0_a)]


mu_0_b = init_state_b
sigma_0_b = 0.01 * np.eye(4, 4)

Bel_b = [(mu_0_b, sigma_0_b)]

x_afterassociative_a = []
y_afterassociative_a = []
x_afterassociative_b = []
y_afterassociative_b = [] 
# x_afterassociative_a.append(init_state_a[0])
# y_afterassociative_a.append(init_state_a[1])
# x_afterassociative_b.append(init_state_b[0])
# y_afterassociative_b.append(init_state_b[1])

for t in range(T):
    # ------------------------------------------------------------------------------------------------------------------------------------
    # data association part
    # prediction
    # print(observation_table[t][0])
    obs_a, obs_b = np.array([observation_table[t][0][0,0],observation_table[t][0][1,0]]), np.array([observation_table[t][1][0,0], observation_table[t][1][1,0]])
    est_a_nxt = np.array([(A(delta_t).dot(Bel_a[t][0]) + B(delta_t).dot(U(Omega_a, t)))[0,0], (A(delta_t).dot(Bel_a[t][0]) + B(delta_t).dot(U(Omega_a, t)))[1,0]])
    est_b_nxt = np.array([(A(delta_t).dot(Bel_b[t][0]) + B(delta_t).dot(U(Omega_b, t)))[0,0], (A(delta_t).dot(Bel_b[t][0]) + B(delta_t).dot(U(Omega_b, t)))[1,0]])
    # est_cov_a = A(delta_t).dot(Bel_a[t][1]).dot(A(delta_t).T) + R
    # est_cov_b = A(delta_t).dot(Bel_b[t][1]).dot(A(delta_t).T) + R
    # linear assignment
    distmatrix = np.zeros((2,2)) # fixed for now
    # print(obs_a, est_a_nxt)
    # print(obs_b, est_b_nxt)
    distmatrix[0][0] = np.linalg.norm(obs_a-est_a_nxt)
    distmatrix[0][1] = np.linalg.norm(obs_a-est_b_nxt)
    distmatrix[1][0] = np.linalg.norm(obs_b-est_a_nxt)
    distmatrix[1][1] = np.linalg.norm(obs_b-est_b_nxt)
    
    # hungarian algorithm
    minr_1 = min(distmatrix[0][0], distmatrix[0][1])
    minr_2 = min(distmatrix[1][0], distmatrix[1][1])
    distmatrix[0][0] -= minr_1
    distmatrix[0][1] -= minr_1
    distmatrix[1][0] -= minr_2
    distmatrix[1][1] -= minr_2
    pairing = (-1,-1)
    if distmatrix[0][0]==0 and distmatrix[1][1]==0:
        pairing = (0,1)
    elif distmatrix[1][0]==0 and distmatrix[0][1]==0:
        pairing = (1,0)
    else:
        minc_1 = min(distmatrix[0][0], distmatrix[1][0])
        minc_2 = min(distmatrix[0][1], distmatrix[1][1])
        distmatrix[0][0] -= minc_1
        distmatrix[1][0] -= minc_1
        distmatrix[0][1] -= minc_2
        distmatrix[1][1] -= minc_2
        if distmatrix[0][0]==0 and distmatrix[1][1]==0:
            pairing = (0,1)
        else:
            pairing = (1,0)
    
    (a,b)  = pairing
    obs_av = obs_a
    obs_bv = obs_b
    if a==1 and b==0:
        obs_av = obs_b
        obs_bv = obs_a 
    # ------------------------------------------------------------------------------------------------------------------------------------
    # print([obs_av[0]], [x_obs_a[t]])
    # print([obs_av[1]], [y_obs_a[t]])
    x_afterassociative_a.append(obs_av[0])
    y_afterassociative_a.append(obs_av[1])
    x_afterassociative_b.append(obs_bv[0])
    y_afterassociative_b.append(obs_bv[1])
    mu_a, sigma_a = kalman_filter(Bel_a[t][0], Bel_a[t][1], np.array([[obs_av[0]], [obs_av[1]]]), A(delta_t), U(Omega_a, t), B(delta_t), R_a, C_a, Q_a)
    Bel_a.append((mu_a, sigma_a))
    mu_b, sigma_b = kalman_filter(Bel_b[t][0], Bel_b[t][1], np.array([[obs_bv[0]], [obs_bv[1]]]), A(delta_t), U(Omega_b, t), B(delta_t), R_b, C_b, Q_b)
    Bel_b.append((mu_b, sigma_b))

predicted_state_a = [Bel_a[i][0] for i in range(len(Bel_a))]
predicted_state_a = np.array(predicted_state_a)

predicted_state_b = [Bel_b[i][0] for i in range(len(Bel_b))]
predicted_state_b = np.array(predicted_state_b)

fig, ax = plt.subplots(1, 1, figsize = (10, 10))
ax.set_xticks(np.arange(x_min_f-10, x_max_f+10, 10))
ax.set_yticks(np.arange(y_min_f-10, y_max_f+10, 10))
ax.set_xlim([x_min_f-10, x_max_f+10])
ax.set_ylim([y_min_f-10, y_max_f+10])
marker = markers.MarkerStyle(marker='s')

line1, = ax.plot(x_motion_a, y_motion_a, color='g')
line2, = ax.plot(x_obs_a, y_obs_a, color='r')
line3, = ax.plot(predicted_state_a[0, 0, 0], predicted_state_a[0, 1, 0], color='b')

line4, = ax.plot(x_motion_b, y_motion_b, color='g')
line5, = ax.plot(x_obs_b, y_obs_b, color='r')
line6, = ax.plot(predicted_state_b[0, 0, 0], predicted_state_b[0, 1, 0], color='b')

line7, = ax.plot(x_afterassociative_a, y_afterassociative_a, color = 'c')
line8, = ax.plot(x_afterassociative_b, y_afterassociative_b, color = 'c')

scat1 = plt.scatter([x_motion_a[0]], [y_motion_a[0]], c='g', s=50, edgecolors='k')
scat2 = plt.scatter([x_obs_a[0]], [y_obs_a[0]], c='r', s=50, edgecolors='k', marker=marker)
scat3 = plt.scatter([predicted_state_a[0, 0, 0]], [predicted_state_a[0, 1, 0]], c='b', s=50, edgecolors='k')

scat4 = plt.scatter([x_motion_b[0]], [y_motion_b[0]], c='g', s=50, edgecolors='k')
scat5 = plt.scatter([x_obs_b[0]], [y_obs_b[0]], c='r', s=50, edgecolors='k', marker = marker)
scat6 = plt.scatter([predicted_state_b[0, 0, 0]], [predicted_state_b[0, 1, 0]], c='b', s=50, edgecolors='k')

scat7 = plt.scatter([x_afterassociative_a[0]], [y_afterassociative_a[0]], c='c', s=50 , edgecolors='k')
scat8 = plt.scatter([x_afterassociative_b[0]], [y_afterassociative_b[0]], c='c', s=50, edgecolors='k')

line1.set_label('Actual motion a')
line2.set_label('Observed motion a')
line3.set_label('Predicted motion a')


line4.set_label('Actual motion b')
line5.set_label('Observed motion b')
line6.set_label('Predicted motion b')

line7.set_label('data associative a')
line8.set_label('data associative b')

plt.grid()
plt.legend(loc="upper left")

def animate(i):
    line1.set_data(x_motion_a[:i+1], y_motion_a[:i+1])
    line2.set_data(x_obs_a[:i+1], y_obs_a[:i+1])
    line3.set_data(predicted_state_a[:i+1, 0, 0], predicted_state_a[:i+1, 1, 0])
    line4.set_data(x_motion_b[:i+1], y_motion_b[:i+1])
    line5.set_data(x_obs_b[:i+1], y_obs_b[:i+1])
    line6.set_data(predicted_state_b[:i+1, 0, 0], predicted_state_b[:i+1, 1, 0])
    line7.set_data(x_afterassociative_a[:i+1], y_afterassociative_a[:i+1])
    line8.set_data(x_afterassociative_b[:i+1], y_afterassociative_b[:i+1])
    scat1.set_offsets(np.c_[[x_motion_a[i]], [y_motion_a[i]]])
    scat2.set_offsets(np.c_[[x_obs_a[i]], [y_obs_a[i]]])
    scat3.set_offsets(np.c_[[predicted_state_a[i, 0, 0]], [predicted_state_a[i, 1, 0]]])
    scat4.set_offsets(np.c_[[x_motion_b[i]], [y_motion_b[i]]])
    scat5.set_offsets(np.c_[[x_obs_b[i]], [y_obs_b[i]]])
    scat6.set_offsets(np.c_[[predicted_state_b[i, 0, 0]], [predicted_state_b[i, 1, 0]]])
    scat7.set_offsets(np.c_[[x_afterassociative_a[i]], [y_afterassociative_a[i]]])
    scat8.set_offsets(np.c_[[x_afterassociative_b[i]], [y_afterassociative_b[i]]])
    return line1, line2, line3, line4, line5, line6, line7, line8, scat1, scat2, scat3, scat4, scat5, scat6, scat7, scat8,

ani = animation.FuncAnimation(fig, animate, T, interval=delta_t * 1000, blit=True)
ani.save('i.mp4')

plt.show()
