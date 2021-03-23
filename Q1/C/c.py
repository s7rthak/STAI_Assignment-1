from itertools import repeat
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import numpy as np
import matplotlib.markers as markers

T=30

# initializing the probability matrix for bayes filter P[i,j,k]
# i:time j:x-axis location k:y-axis location
P = np.zeros((T+1,30,30))
P_fwd = np.zeros((T+1,30,30))
for i in range(30):
    for j in range(30):
        P_fwd[0,i,j] = 1/900


def chebyshev(a, b):
    return max(abs(a[0] - b[0]),abs(a[1] - b[1]))

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
    movement_direction = random.choices(choices, action_weights, k=1)[0]
    return movement_direction

def random_sample(arr: np.array, size: int = 1) -> np.array:
    return arr[np.random.choice(len(arr), size=size, replace=False)]

# Robot motion model initialisation.
motion_model = { 'up': 0.4, 'down': 0.1, 'left': 0.2, 'right': 0.3 }
x, y = random.randrange(30), random.randrange(30)
x, y = 15, 15
robot_bf = Robot(x, y)

# Sensor initialisation.
sensor_1, sensor_2, sensor_3, sensor_4 = Sensor(8, 15), Sensor(15, 15), Sensor(22, 15), Sensor(15, 22)
all_sensors = [sensor_1, sensor_2, sensor_3, sensor_4]
all_measure(all_sensors, robot_bf)

# Running the simulation.
for i in range(T):
    robot_bf.next_state(transition_model(robot_bf, motion_model))
    all_measure(all_sensors, robot_bf)
    # calculating probability of a state using baye's filter
    P_dash = np.zeros((30,30))
    for a in range(30):
        for b in range(30):
            for action in motion_model:
                if action == 'up' and b>0:
                    P_dash[a,b] += motion_model[action] * P_fwd[i,a,b-1]
                if action == 'down' and b<29:
                    P_dash[a,b] += motion_model[action] * P_fwd[i,a,b+1]
                if action == 'left' and a<29:
                    P_dash[a,b] += motion_model[action] * P_fwd[i,a+1,b]
                if action == 'right' and a>0:
                    P_dash[a,b] += motion_model[action] * P_fwd[i,a-1,b]
    eta=0
    for a in range(30):
        for b in range(30):
            for sensor in all_sensors:
                prob_detection = max(0.4, 0.9 - chebyshev((a, b), (sensor.x, sensor.y))/10)
                # making sure probability remains zero outside the distribution
                if prob_detection == 0.4:
                    prob_detection = 0
                if sensor.measurements[i]:
                    P_fwd[i+1,a,b] += prob_detection * P_dash[a,b]
                else:
                    P_fwd[i+1,a,b] += (1-prob_detection) * P_dash[a,b]
            eta += P_fwd[i+1,a,b]

    for a in range(30):
        for b in range(30):
            P_fwd[i+1,a,b] = P_fwd[i+1,a,b]/eta
    
# backward algorithm

# initializinf P_bwd same ad P_fwd for backward pass
P_bwd = np.zeros((T+1,30,30))
for i in range(30):
    for j in range(30):
        P_bwd[0,i,j] = 1/900

# setting up initial bias for end states
for i in range(T):
    eta = 0
    for a in range(30):
        for b in range(30):
            for action in motion_model:
                if action == 'up' and b>0:
                    P_bwd[i+1,a,b] += motion_model[action] * P_bwd[i,a,b-1]
                if action == 'down' and b<29:
                    P_bwd[i+1,a,b] += motion_model[action] * P_bwd[i,a,b+1]
                if action == 'left' and a<29:
                    P_bwd[i+1,a,b] += motion_model[action] * P_bwd[i,a+1,b]
                if action == 'right' and a>0:
                    P_bwd[i+1,a,b] += motion_model[action] * P_bwd[i,a-1,b]
            eta += P_bwd[i+1,a,b]
    

for i in range(min(30,T)):    
    for a in range(30):
        for b in range(30):
            P_bwd[i+1,a,b] = P_bwd[i+1,a,b]/eta
            # unsetting bias for all states excluding end ones
            P_bwd[i,a,b] = 0

if T>30:
    for a in range(30):
        for b in range(30):
            P_bwd[T,a,b] = P_bwd[30,a,b]/eta
            # unsetting bias for all states excluding end ones
            P_bwd[15,a,b] = 0



# starting backpropagation algorithm
for i in range(T-1, -1, -1):
    eta=0
    for a in range(30):
        for b in range(30):
            for action in motion_model:
                if action == 'up' and b<29:
                    for sensor in all_sensors:
                        prob_detection = max(0.4, 0.9 - chebyshev((a, b), (sensor.x, sensor.y))/10)
                        if prob_detection == 0.4:
                            prob_detection = 0.0
                        if sensor.measurements[i]:
                            P_bwd[i,a,b] += prob_detection * P_bwd[i+1,a,b+1] * motion_model[action]
                        else:
                            P_bwd[i,a,b] += 1/900 * P_bwd[i+1,a,b+1] * motion_model[action]
                if action == 'down' and b>0:
                    for sensor in all_sensors:
                        prob_detection = max(0.4, 0.9 - chebyshev((a, b), (sensor.x, sensor.y))/10)
                        if prob_detection == 0.4:
                            prob_detection = 0.0
                        if sensor.measurements[i]:
                            P_bwd[i,a,b] += prob_detection * P_bwd[i+1,a,b-1] * motion_model[action]
                        else:
                            P_bwd[i,a,b] += 1/900 * P_bwd[i+1,a,b-1] * motion_model[action]
                if action == 'left' and a>0:
                    for sensor in all_sensors:
                        prob_detection = max(0.4, 0.9 - chebyshev((a, b), (sensor.x, sensor.y))/10)
                        if prob_detection == 0.4:
                            prob_detection = 0.0
                        if sensor.measurements[i]:
                            P_bwd[i,a,b] += prob_detection * P_bwd[i+1,a-1,b] * motion_model[action]
                        else:
                            P_bwd[i,a,b] += 1/900 * P_bwd[i+1,a-1,b] * motion_model[action]
                if action == 'right' and a<29:
                    for sensor in all_sensors:
                        prob_detection = max(0.4, 0.9 - chebyshev((a, b), (sensor.x, sensor.y))/10)
                        if prob_detection == 0.4:
                            prob_detection = 0.0
                        if sensor.measurements[i]:
                            P_bwd[i,a,b] += prob_detection * P_bwd[i+1,a+1,b] * motion_model[action]
                        else:
                            P_bwd[i,a,b] += 1/900 * P_bwd[i+1,a+1,b] * motion_model[action]
            eta += P_bwd[i,a,b]
    
    for a in range(30):
        for b in range(30):
            P_bwd[i,a,b] = P_bwd[i,a,b]/eta
            # P_bwd[i,a,b] = 10*P_bwd[i,a,b]

for i in range(T+1):
    eta=0
    # print(P_fwd[i,:,:])
    # print(P_bwd[i,:,:])
    for a in range(30):
        for b in range(30):
            P[i,a,b] = P_fwd[i,a,b] * P_bwd[i,a,b]
            eta += P[i,a,b]

    for a in range(30):
        for b in range(30):
            P[i,a,b] = P[i,a,b]/eta





 
    





# Simulating the robot motion through grid motion.
fig = plt.figure(figsize=(10, 10))
ax = fig.gca()
ax.set_xticks(np.arange(0, 31, 1))
ax.set_yticks(np.arange(0, 31, 1))
# made sure the points lie inside coordinate axes rather than on coordinate axes
ax.set_xlim([-0.5, 29.5])
ax.set_ylim([-0.5, 29.5])
cmap = colors.ListedColormap(['b', 'g', 'r'])
scatter_color = [1 if s.measurements[0] else 2 for s in all_sensors]
scatter_color.append(0)
marker = markers.MarkerStyle(marker='s')
scat = plt.scatter([x, sensor_1.x, sensor_2.x, sensor_3.x, sensor_4.x], [y, sensor_1.y, sensor_2.y, sensor_3.y, sensor_4.y], c=scatter_color, s=200, cmap='Greys', edgecolors='k', marker=marker)
scat2 = plt.scatter([sensor_1.x, sensor_2.x, sensor_3.x, sensor_4.x, x], [sensor_1.y, sensor_2.y, sensor_3.y, sensor_4.y, y], c=scatter_color,cmap=cmap, s=200, edgecolors='k', marker=marker)
# scat3 = plt.scatter([x], [y], s=200, c=0, cmap=cmap, edgecolors='k')
scat4 = plt.scatter([], [], c='m', s=200, edgecolors='k', marker=marker)

global last_best
last_best = (-1,-1)



# Handling how frames are updated in animation.
def update_plot(i):
    global last_best
    arr = []
    brr = []
    colmap = []
    # finding point with maximum probablity
    result = np.where(P[i,:,:] == np.amax(P[i,:,:]))
    list_of_coordinates = np.array(list(zip(result[0], result[1])))
    minval = 3000
    # print(last_best)
    if last_best == (-1,-1):
        last_best = (random_sample(list_of_coordinates)[0,0],random_sample(list_of_coordinates)[0,1]) 
    else:
        current_best = random_sample(list_of_coordinates)
        for coordinates in list_of_coordinates:
            a,b = last_best
            c,d = coordinates
            val = np.linalg.norm([d-b,c-a])
            if val < minval:
                minval = val
                current_best = (c,d)
        last_best = current_best
    
    # print(last_best)
    a_est, b_est = last_best
    
    for a in range(30):
        for b in range(30):
            arr.append(a)
            brr.append(b)
            colmap.append(P[i,a,b]*1000)
    
    sensor_pos_x, sensor_pos_y = [], []
    sensor_cmap = []
    for s in all_sensors:
        sensor_pos_x.append(s.x)
        sensor_pos_y.append(s.y)
        if s.measurements[i]:
            sensor_cmap.append(1)
        else:
            sensor_cmap.append(2)
    
    sensor_pos_x.append(robot_bf.movement_history[i][0])
    sensor_pos_y.append(robot_bf.movement_history[i][1])
    sensor_cmap.append(0)

    scat.set_offsets(np.c_[arr, brr])
    scat.set_array(np.array(colmap))
    scat2.set_offsets(np.c_[sensor_pos_x, sensor_pos_y])
    scat2.set_array(np.array(sensor_cmap))
    scat4.set_offsets(np.c_[[a_est], [b_est]])
    return scat, scat2, scat4,

plt.grid()
ani = animation.FuncAnimation(fig, update_plot, frames=range(len(robot_bf.movement_history)), interval=500, repeat=False, blit=True)
ani.save('c.mp4')
plt.show()
