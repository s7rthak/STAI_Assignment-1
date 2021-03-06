from itertools import repeat
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.markers as markers
import numpy as np

def chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

def all_measure(all_sensors, robot_bf):
    for sensor in all_sensors:
        sensor.measure(robot_bf)

class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.movement_history = [(x, y)]
    def next_state(self, action):
        if action == 'up':
            self.y += 1
        elif action == 'down':
            self.y -= 1
        elif action == 'left':
            self.x -= 1
        elif action == 'right':
            self.x += 1
        self.movement_history.append((self.x, self.y))

class Sensor:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.measurements = []
    def measure(self, robot_bf):
        prob_detection = max(0.4, 0.9 - chebyshev((robot_bf.x, robot_bf.y), (self.x, self.y))/10)
        # making sure probability remains zero outside the distribution
        if prob_detection == 0.4:
            prob_detection = 0
        measurement = np.random.choice([True, False], 1, p=[prob_detection, 1-prob_detection])[0]
        self.measurements.append(measurement)

# This function returns the next stochastic action.
def transition_model(robot_bf, motion_model):
    choices = ['up', 'down', 'left', 'right']
    edge_corner = False
    if robot_bf.x == 0:
        choices.remove('left')
        edge_corner = True
    if robot_bf.x == 29:
        choices.remove('right')
        edge_corner = True
    if robot_bf.y == 0:
        choices.remove('down')
        edge_corner = True
    if robot_bf.y == 29:
        choices.remove('up')
        edge_corner = True
    action_weights = None
    if not edge_corner:
        action_weights = [motion_model[motion] for motion in choices]
    else:
        action_weights = [1 for motion in choices]
    return random.choices(choices, action_weights, k=1)[0]

# Robot motion model initialisation.
motion_model = { 'up': 0.4, 'down': 0.1, 'left': 0.2, 'right': 0.3 }
x, y = random.randrange(30), random.randrange(30)
robot_bf = Robot(x, y)

# Sensor initialisation.
sensor_1, sensor_2, sensor_3, sensor_4 = Sensor(8, 15), Sensor(15, 15), Sensor(22, 15), Sensor(15, 22)
all_sensors = [sensor_1, sensor_2, sensor_3, sensor_4]
all_measure(all_sensors, robot_bf)

T = 100
# Running the simulation.
for i in range(T):
    robot_bf.next_state(transition_model(robot_bf, motion_model))
    all_measure(all_sensors, robot_bf)   

# Simulating the robot motion through grid motion.
fig = plt.figure(figsize=(10, 10))
ax = fig.gca()
ax.set_xticks(np.arange(0, 31, 1))
ax.set_yticks(np.arange(0, 31, 1))
ax.set_xlim([-0.5, 29.5])
ax.set_ylim([-0.5, 29.5])
cmap = colors.ListedColormap(['b', 'g', 'r'])
scatter_color = [1 if s.measurements[0] else 2 for s in all_sensors]
scatter_color.insert(0, 0)

P = np.arange(30)
all_points = np.dstack(np.meshgrid(P, P)).reshape(-1, 2)
marker = markers.MarkerStyle(marker='s')
scat2 = plt.scatter(all_points[:, 0], all_points[:, 1], c=np.zeros(900, dtype=int), s=200, cmap='Greys', edgecolors='k', marker=marker)
scat = plt.scatter([sensor_1.x, sensor_2.x, sensor_3.x, sensor_4.x, x], [sensor_1.y, sensor_2.y, sensor_3.y, sensor_4.y, y], c=scatter_color,cmap=cmap, s=200, edgecolors='k', marker=marker)

# Handling how frames are updated in animation.
def update_plot(i):
    col_map = [1 if s.measurements[i] else 2 for s in all_sensors]
    col_map.insert(0, 0)
    scat.set_offsets(np.c_[[robot_bf.movement_history[i][0], sensor_1.x, sensor_2.x, sensor_3.x, sensor_4.x], [robot_bf.movement_history[i][1], sensor_1.y, sensor_2.y, sensor_3.y, sensor_4.y]])
    scat.set_array(np.array(col_map))
    return scat, 

plt.grid()
ani = animation.FuncAnimation(fig, update_plot, frames=range(len(robot_bf.movement_history)), interval=500, repeat=False, blit=True)
ani.save('a.mp4')
plt.show()
    