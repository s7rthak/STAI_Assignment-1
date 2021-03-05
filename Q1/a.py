from itertools import repeat
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import numpy as np

def manHattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

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
        prob_detection = max(0, 0.9 - manHattan((robot_bf.x, robot_bf.y), (self.x, self.y))/10)
        measurement = np.random.choice([True, False], 1, p=[prob_detection, 1-prob_detection])[0]
        self.measurements.append(measurement)

    
def transition_model(robot_bf, motion_model):
    choices = ['up', 'down', 'left', 'right']
    if robot_bf.x == 0:
        choices.remove('left')
    if robot_bf.x == 29:
        choices.remove('right')
    if robot_bf.y == 0:
        choices.remove('down')
    if robot_bf.y == 29:
        choices.remove('up')
    action_weights = [motion_model[motion] for motion in choices]
    return random.choices(choices, action_weights, k=1)[0]

# Robot motion model initialisation.
motion_model = { 'up': 0.4, 'down': 0.1, 'left': 0.2, 'right': 0.3 }
x, y = random.randrange(30), random.randrange(30)
robot_bf = Robot(x, y)

# Sensor initialisation.
sensor_1, sensor_2, sensor_3, sensor_4 = Sensor(8, 15), Sensor(15, 15), Sensor(22, 15), Sensor(15, 22)
all_sensors = [sensor_1, sensor_2, sensor_3, sensor_4]
all_measure(all_sensors, robot_bf)

# Running the simulation.
for i in range(30):
    robot_bf.next_state(transition_model(robot_bf, motion_model))
    all_measure(all_sensors, robot_bf)   

# Simulating the robot motion.
fig = plt.figure(figsize=(10, 10))
ax = fig.gca()
ax.set_xticks(np.arange(0, 31, 1))
ax.set_yticks(np.arange(0, 31, 1))
ax.set_xlim([-0.5, 29.5])
ax.set_ylim([-0.5, 29.5])
cmap = colors.ListedColormap(['b', 'g', 'r'])
scatter_color = [1 if s.measurements[0] else 2 for s in all_sensors]
scatter_color.insert(0, 0)
scat = plt.scatter([x, sensor_1.x, sensor_2.x, sensor_3.x, sensor_4.x], [y, sensor_1.y, sensor_2.y, sensor_3.y, sensor_4.y], c=scatter_color, s=200, cmap=cmap)

def update_plot(i):
    col_map = [1 if s.measurements[i] else 2 for s in all_sensors]
    col_map.insert(0, 0)
    scat.set_offsets(np.c_[[robot_bf.movement_history[i][0], sensor_1.x, sensor_2.x, sensor_3.x, sensor_4.x], [robot_bf.movement_history[i][1], sensor_1.y, sensor_2.y, sensor_3.y, sensor_4.y]])
    scat.set_array(np.array(col_map))
    return scat, 

plt.grid()
ani = animation.FuncAnimation(fig, update_plot, frames=range(len(robot_bf.movement_history)), interval=500, repeat=False, blit=False)
ani.save('a.gif')
plt.show()
    