from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.lib.function_base import interp
import math
import matplotlib.markers as markers
import matplotlib.patches as pat
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

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

def _plot_gaussian(mean, covariance, ax, color='k', zorder=0):
    """Plots the mean and 2-std ellipse of a given Gaussian"""
    # plt.plot(mean[0], mean[1], color[0] + ".", zorder=zorder)

    if covariance.ndim == 1:
        covariance = np.diag(covariance)

    radius = np.sqrt(5.991)
    eigvals, eigvecs = np.linalg.eig(covariance)
    axis = np.sqrt(eigvals) * radius
    slope = eigvecs[1][0] / eigvecs[1][1]
    angle = 180.0 * np.arctan(slope) / np.pi

    ell = pat.Ellipse(
        mean, axis[0], axis[1], angle=angle,
        fill=False, color=color, linewidth=1, zorder=zorder, animated=True
    )
    return ax.add_patch(ell)

# -------------------------------------------------------------------------------------------------------------------
# taken from https://github.com/yugaro/machine-learning/tree/444db199d5f66d63c4bb60cb367f7b774c5cef7e/Bayesian-CBF


def angle_from_rotmat(R):
    """
    >>> theta = np.random.rand() * 2*np.pi - np.pi
    >>> thetae = angle_from_rotmat(rotmat2D(theta))
    >>> np.allclose(thetae, theta)
    True
    """
    return np.arctan2(R[1, 0], R[0, 0])

def var_to_scale_theta(V):
    """
    >>> scale = np.abs(np.random.rand(2)) * 10
    >>> theta = np.random.rand() * (2*np.pi) - np.pi
    >>> s, t = var_to_scale_theta(scale_theta_to_var(scale, theta))
    >>> allclose = partial(np.allclose, rtol=1e-2, atol=1e-5)
    >>> allclose(s, scale)
    True
    >>> allclose(t, theta)
    True
    """
    w, E = np.linalg.eig(V)
    scale = 3*w
    theta = angle_from_rotmat(E)
    return scale, theta


def draw_ellipse(ax, scale, theta, x0, **kwargs):
    ellipse = Ellipse((0,0),
                      width=2*scale[0],
                      height=2*scale[1],
                      fill=False, color='b',
                      **kwargs)

    transf = transforms.Affine2D() \
        .rotate_deg(theta * 180/np.pi) \
        .translate(x0[0], x0[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse), ellipse

def demo_plot_ellipse(V, mean, ax):
    ellipse = draw_ellipse(ax, *var_to_scale_theta(V), mean)[1]
    return ellipse

# ------------------------------------------------------------------------------------------------------------------

U = np.zeros((2, 1))
C = np.eye(2, 4)
R = np.zeros((4, 4))
R[0, 0] = 1
R[1, 1] = 1
R[2, 2] = 0.0001
R[3, 3] = 0.0001
Q = 10 * np.eye(2, 2)

init_location = 10 * np.ones((2, 1))
init_velocity = np.ones((2, 1))

init_state = np.vstack((init_location, init_velocity))
my_airplane = Airplane(init_state)
my_airplane.observed(observation_model(my_airplane.s, C, np.random.multivariate_normal(np.zeros(2,), Q).reshape(2, 1)))

T = 20
delta_t = 1

for t in range(T):
    S = motion_model(my_airplane.s, A(delta_t), B(delta_t), U, np.random.multivariate_normal(np.zeros(4,), R).reshape(4, 1))
    my_airplane.transition(S)
    my_airplane.observed(observation_model(my_airplane.s, C, np.random.multivariate_normal(np.zeros(2,), Q).reshape(2, 1)))

x_motion = [my_airplane.state_history[j][0, 0] for j in range(T)]
y_motion = [my_airplane.state_history[j][1, 0] for j in range(T)]
x_obs = [my_airplane.observation[j][0, 0] for j in range(T)]
y_obs = [my_airplane.observation[j][1, 0] for j in range(T)]

mu_0 = init_state
sigma_0 = 0.0001 * np.eye(4, 4)

Bel = [(mu_0, sigma_0)]

for t in range(T):
    mu, sigma = kalman_filter(Bel[t][0], Bel[t][1], np.array([[x_obs[t]], [y_obs[t]]]), A(delta_t), U, B(delta_t), R, C, Q)
    Bel.append((mu, sigma))

predicted_state = [Bel[i][0] for i in range(len(Bel))]
predicted_state = np.array(predicted_state)

fig, ax = plt.subplots(1, 1, figsize = (10, 10))
ax.set_xticks(np.arange(0, 50, 1))
ax.set_yticks(np.arange(0, 50, 1))
ax.set_xlim([0, 50])
ax.set_ylim([0, 50])
# ell = [_plot_gaussian(mu_0[:2, 0], sigma_0)]

# for t in range(1, 20):
#     ell.append(_plot_gaussian(Bel[t][0][:2, 0], Bel[t][1]))

# e = ell[0]
# ax.add_patch(e)
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

last_ell = None

def animate(i):
    line1.set_data(x_motion[:i+1], y_motion[:i+1])
    line2.set_data(x_obs[:i+1], y_obs[:i+1])
    line3.set_data(predicted_state[:i+1, 0, 0], predicted_state[:i+1, 1, 0])
    ell = demo_plot_ellipse(Bel[i][1][:2, :2], Bel[i][0][:2], ax)
    global last_ell
    if last_ell:
        last_ell.set_visible(False)
    last_ell = ell
    scat1.set_offsets(np.c_[[x_motion[i]], [y_motion[i]]])
    scat2.set_offsets(np.c_[[x_obs[i]], [y_obs[i]]])
    scat3.set_offsets(np.c_[[predicted_state[i, 0, 0]], [predicted_state[i, 1, 0]]])
    return line1, line2, line3, scat1, scat2, scat3, 

ani = animation.FuncAnimation(fig, animate, T, interval=delta_t * 1000, blit=False)
ani.save('c.mp4')

plt.show()