import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## order down-left, top-left, down-right, top-right
door1_inside_frame = [[184.5, 400, 0],
                      [184.5, 400, 234.5],
                      [306.5, 400, 0],
                      [306.5, 400, 234.5]]

door1_outside_frame = [[180.5, 400, 0],
                      [180.5, 400, 238],
                      [310.5, 400, 0],
                      [310.5, 400, 238]]

door1_handle = [[191.5, 400, 108.5]]

# left window
window1_frame = [[-155, 400, 105],
                  [-155, 400, 174],
                  [-27, 400, 105],
                  [-27, 400, 174]]

window2_frame = [[-21.5, 400, 105],
                  [-21.5, 400, 174],
                  [106.5, 400, 105],
                  [106.5, 400, 174]]


two_window_outise_frame = [[-160.5, 400, 99.5],
                           [-160.5, 400, 179.5],
                            [114, 400, 99.5],
                           [114, 400, 179.5]]
print(type(door1_outside_frame))

# door 2
door2_outside_frame = [[320.5, 243, 0],
                       [320.5, 243, 243.5],
                       [320.5, 101, 0],
                       [320.5, 101, 243.5]]
door2_inside_frame = [[320.5, 238, 0],
                       [320.5, 238, 243.5],
                       [320.5, 106, 0],
                       [320.5, 106, 243.5]]

door2_depth = [[324.5, 238, 0],
                       [324.5, 238, 243.5],
                       [324.5, 106, 0],
                       [324.5, 106, 243.5]]
# handle of door 2
door2_handle = [[320.5, 110, 125]]

# switch of door 2
door2_switch = [[320.5, 93, 141],
                [320.5, 93, 166],
                [320.5, 86.5, 141],
                [320.5, 86.5, 166]]


# fire blanket box
fire_blanket_box = [[320.5, 270.5, 137],
                    [320.5, 270.5, 162],
                    [320.5, 249, 137],
                    [320.5, 249, 162]]

# working bench
working_bench = [[255.5, -94, 0],
                 [255.5, -94, 92],
                 [320.5, -94, 0],
                 [320.5, -94, 92],
                 [320.5, -94, 147],

                 [255.5, -301.5, 0],
                 [255.5, -301.5, 92],
                 [320.5, -301.5, 0],
                 [320.5, -301.5, 92],
                 [320.5, -301.5, 147]
                 ]


# Patslide

patslide = [[-200, 147, 27.5],
            [-200, 147, 180],
            [-200, 210.5, 27.5],
            [-200, 210.5, 180]]

# screen
wall_screen = [[-200, 244, 155.5],
               [-200, 244, 185.5],
               [-200, 286.5, 155.5],
               [-200, 286.5, 185.5]]


# lamp
lamp = [[-200, 233, 213]]

fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(111, projection='3d')
# ax.set_aspect('equal')
def plot_object(lst):
    arr = np.array(lst)
    ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2])
    return None

plot_object(working_bench)
plot_object(door2_inside_frame)
plot_object(door2_outside_frame)
plot_object(door2_depth )
plot_object(door2_handle )
plot_object(door2_switch )
plot_object(fire_blanket_box)
plot_object(door1_inside_frame)
plot_object(door1_outside_frame)
plot_object(door1_handle)

plot_object(window1_frame)
plot_object(window2_frame)
plot_object(two_window_outise_frame)

plot_object(patslide)
plot_object(lamp)
plot_object(wall_screen)
# ax.scatter(getXYZ(door2_switch))
# ax.scatter(getXYZ(fire_blanket_box))
plt.show()


## 少了window depth 还有button的relative distance