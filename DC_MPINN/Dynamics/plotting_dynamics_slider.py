import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
import csv
import os

filename="dynamic_displacement_v1.csv"
data=np.genfromtxt(filename,delimiter=",")
shape=np.shape(data)

num_points=5
num_times=5
duration=3

fig = plt.figure()
plt.subplots_adjust(bottom=0.25)
ax2 = fig.add_subplot( projection='3d')

len_row=shape[1]
num_files=shape[0]//4
timestep=duration/(num_times-1)

file=0
ax2.scatter(data[file+1,0::num_times],data[file+2,0::num_times],data[file+3,0::num_times], color='Blue')
time_slider = plt.axes([0.3, 0.05, 0.65, 0.03])
time = Slider(time_slider, 'Time', 0, duration*num_files, valinit=0,valstep=timestep)


ax2.set_xlabel('X position')
ax2.set_ylabel('Y position')
ax2.set_zlabel('Z position')


ax2.set_zlim(0, 1.25)
ax2.set_xlim(0, 2.5)
ax2.set_ylim(0, 1.25)

def update(val):
    i = time.val


    file=int(np.floor((i+.001)/duration)) #at aux values, take the higher file number (constant value is just for stability))
    sub_time=int(((i-file*duration)/duration)*(num_times-1))
    if file==num_files:
        file=num_files-1
        sub_time=num_times-1
    print(i)
    print(sub_time)
    print(file)
    ax2.clear()
    ax2.scatter(data[4*file+1,sub_time::num_times],data[4*file+2,sub_time::num_times],data[4*file+3,sub_time::num_times], color='blue')
    ax2.set_zlim(0, 1.25)
    ax2.set_xlim(0, 2.5)
    ax2.set_ylim(0, 1.25)
    fig.canvas.draw_idle()

time.on_changed(update)



plt.show()