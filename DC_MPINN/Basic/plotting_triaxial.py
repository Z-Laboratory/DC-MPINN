import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
import csv
import os

analytic_filename="triaxial.csv"
PINN_filename="triaxial_pinn.csv"

data=np.genfromtxt(analytic_filename,delimiter=",")

data=data[1:,:]

PINN_results=np.genfromtxt(PINN_filename,delimiter=",")

PINN_results=PINN_results[1:,:]


pinn_s1=PINN_results[:,0]
pinn_s2=PINN_results[:,1]
pinn_s3=PINN_results[:,2]
pinn_x_max=PINN_results[:,3]
pinn_y_max=PINN_results[:,4]
pinn_z_max=PINN_results[:,5]




fig = plt.figure()
plt.subplots_adjust(bottom=0.25)


ax2 = fig.add_subplot( projection='3d')
s1=data[:,0]
s2=data[:,1]
s3=data[:,2]
x_max=data[:,3]
y_max=data[:,4]
z_max=data[:,5]

MSE_x=np.mean(np.square(x_max-pinn_x_max))
MSE_y=np.mean(np.square(y_max-pinn_y_max))
MSE_z=np.mean(np.square(z_max-pinn_z_max))


print("X mse",MSE_x,"Y mse",MSE_y,"Z mse",MSE_z)
print("total MSE",(MSE_x+MSE_y+MSE_z)/3)

num_points=s1.size
a=round(num_points**(1/3))

ax2.scatter(x_max[:], y_max[:], z_max[:], color='blue')
ax2.scatter(pinn_x_max[:], pinn_y_max[:], pinn_z_max[:], color='orange')
ax2.scatter(x_max[0], y_max[0], z_max[0], s=200, color='red')
ax2.set_zlim(0, 2)
ax2.set_xlim(0, 2)
ax2.set_ylim(0, 2)

axstress1 = plt.axes([0.3, 0.25, 0.65, 0.03])
axstress2 = plt.axes([0.3, 0.15, 0.65, 0.03])
axstress3 = plt.axes([0.3, 0.05, 0.65, 0.03])
stress1 = Slider(axstress1, 'Stress1', 0, 1, valinit=0,valstep=1/(a-1))
stress2 = Slider(axstress2, 'Stress2', 0, 1, valinit=0,valstep=1/(a-1))
stress3 = Slider(axstress3, 'Stress3', 0, 1, valinit=0,valstep=1/(a-1))

ax2.set_xlabel('X position')
ax2.set_ylabel('Y position')
ax2.set_zlabel('Z position')

def update(val):
    i = stress1.val*(a-1)
    j = stress2.val*(a-1)
    k = stress3.val*(a-1)
    index=round(a**2*i+a*j+k)
    ax2.clear()
    ax2.scatter(pinn_x_max[:], pinn_y_max[:], pinn_z_max[:], color='orange')
    ax2.scatter(x_max[:],y_max[:],z_max[:],color='blue')
    ax2.scatter(x_max[index],y_max[index],z_max[index],s=200,color='red')
    ax2.scatter(pinn_x_max[index], pinn_y_max[index], pinn_z_max[index],s=50, color='green')
    # ax2.set_zlim(0, 2)
    # ax2.set_xlim(0, 2)
    # ax2.set_ylim(0, 2)
    fig.canvas.draw_idle()

stress1.on_changed(update)
stress2.on_changed(update)
stress3.on_changed(update)


plt.show()