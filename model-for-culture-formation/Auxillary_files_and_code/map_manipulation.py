import scipy.signal
import numpy as np
from graphics import *
import matplotlib.pyplot as plt
import time
import math

iteration_number=17
parameter_list = np.arange(10)
filename_no_conquered = './Simulations/no_conquered_array_' + str(iteration_number)+str(parameter_list)+'.npy'
np.save(filename_no_conquered, parameter_list)

curvature_europe_ini = np.loadtxt('curvature200x200.txt')
curvature_europe = curvature_europe_ini


mountain_europe_ini = np.loadtxt('mountain_background.txt')
mountain_europe = mountain_europe_ini

plt.imshow(mountain_europe, vmin = 0, vmax = 1)
plt.colorbar()
plt.axis('off')
plt.show()

rivers_ini = np.genfromtxt('rivers200x200-curated7.csv', delimiter=';')
rivers_ini = np.reshape(rivers_ini, (200,200))
rivers = np.flip(rivers_ini, axis = 0)

map_europe_ini = np.loadtxt('map200x200_v3.txt')

map_europe = np.flip(map_europe_ini, axis = 0)

    # 2nd axis
n = map_europe_ini.shape[0]

    # 1st axis
m = map_europe_ini.shape[1]

land = np.ones((n,m))


for i in range(0, n):
    for j in range(0, m):
        if map_europe[i, j] == 0 or rivers[i, j] == 1:
            land[i, j] = 0


land  =np.flip(land, axis = 0)

no_land = np.ma.masked_where(land == 1, land)

plt.imshow(no_land, cmap = "Greys")
plt.colorbar()
plt.axis('off')
plt.show()




plt.imshow(mountain_europe, cmap = "inferno", vmin = 0, vmax = 1)
plt.colorbar()
plt.imshow(no_land,cmap = "Greys")
plt.axis('off')
plt.show()





#plt.hist(np.reshape(curvature_europe[curvature_europe!=0], -1), bins="auto")
#plt.show()
# and local defensive bonus
def mountain_background_function(curvature):
    return 1-np.exp(-curvature/100)


mountain_background = mountain_background_function(curvature_europe)
np.savetxt("mountain_background.txt",mountain_background)



#defence_background = defence_background_function(curvature_europe[curvature_europe!=0])


plt.hist(np.reshape(mountain_background, -1), bins="auto")
plt.show()
db = defence_background_function(curvature_europe)
plt.imshow(db, vmin = 0, vmax = 1)
plt.colorbar()
plt.show()
n = 200
m = 200
defence_background = np.zeros((n,m))
for i in range(0,n):
    for j in range(0,m):
    	defence_background[i,j] = defence_background_function(curvature_europe[i, j])
