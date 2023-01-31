import numpy as np
from graphics import *
import os.path
import matplotlib.pyplot as plt

#Parameters:
    #Gives the value of the fluctuations
fluc_size_initial = 0.2

    #Mountain parameters:
mountain_defence_parameter = 2 #The effect of mountains to defense is in [1,2]
mountain_perimeter_parameter = 0.5 #The effect of mountains on perimeter is [0.5,1]

    #Parameters describing the effect of sea/river borders and owning both sides of a river.
sea_border_parameter = 0.5 # Sea_border counts as 0.5
river_area_bonus = 8 # Added to area if controlling both sides of a river.

    # Set mode_neighbours to 'circle' for a circular attacking field
    # set mode_neighbours to 'square' for a square-formed attacking field
mode_neighbours = 'circle'
radius = 4 #Radius of attack is 4.
N=20 # number of interations
start_it = 100 #start iteration
no_era = 10 # number of eras
length_era = 100 #length of an era
parameter_list = [length_era, fluc_size_initial,mountain_defence_parameter,mountain_perimeter_parameter,sea_border_parameter,river_area_bonus,radius]

era = 9
iteration_number = 1


print(os.path.isfile(f'./Simulations2/no_conquered_array_' + str(iteration_number)+str(era)+str(parameter_list)+'.npy'))


#print(os.path.isfile(f"./Simulations/country_array_{(0,0,0.125,0.5,6)}.npy"))
#for i in range(20):
#    arr = np.arange(i)
#    np.save(f"array_number_{i}.npy", arr)
N=8 #number of iterations
firstca = np.load(f'./Simulations/no_conquered_array_' + str(iteration_number)+str(era)+str(parameter_list)+'.npy')

n = np.shape(firstca)[0]
m = np.shape(firstca)[1]





meta_conquered = np.zeros((n,m))
start_it = 100
for k in range(N):
    print(k)
    #f"./Simulations/country_array_{(k,l,0.25,0.5,6)}.npy"
    #no_conquered_array = np.load(f"./Simulations/no_conquered_array_{(k,9,0.125,0.5,6)}.npy")
    no_conquered_array = np.load(f'./Simulations/no_conquered_array_' + str(k + start_it)+str(era)+str(parameter_list)+'.npy')
    for i in range(0,n):
        for j in range(0,m):
            meta_conquered[i,j] +=  min(no_conquered_array[i,j], 30)


#Frequency conversion
arr = np.zeros((3,3))
arr[2,2] = 1
print(arr)
print(arr==0)

meta_conquered = 1/N*meta_conquered

meta_conquered = np.rot90(meta_conquered,2)

meta_conquered = np.flip(meta_conquered,axis =1)



plt.imshow(meta_conquered,cmap ="inferno")

plt.colorbar()


#for i in range(0,n):
#    for j in range(0,m):
#        if cur_country_array[i,j]==0:
#            plt.scatter(i,j)

cur_country_array = np.load(f'./Simulations/no_conquered_array_' + str(k + start_it)+str(era)+str(parameter_list)+'.npy')

nonland = cur_country_array == 0

nonland = np.rot90(nonland,2)

nonland = np.flip(nonland,axis =1)

cur_country_array = np.rot90(cur_country_array, 2)
cur_country_array = np.flip(cur_country_array,axis =1)

masked_data = np.ma.masked_where(cur_country_array != 0, cur_country_array)

plt.imshow(masked_data,  cmap=plt.cm.Blues)

plt.show()
##From here on we plot the borders with the same parameters as in the simulation:
windowsize = 600

win = GraphWin("Borders of Europe", windowsize * m / n, windowsize, autoflush=False)
win.setCoords(0,0,m,n)


rectarray = []
for i in range(0,n):
    for j in range(0,m):
        tmprect = Rectangle(Point(j,i),Point(j+1,i+1))
        tmprect.setOutline(color_rgb(140,140,140))
        tmprect.draw(win)
        rectarray.append(tmprect)
rectarray = np.reshape(rectarray, (n,m))

for i in range(0,n):
    for j in range(0,m):
        if country_array[i,j]==0:
            rectarray[i,j].setFill(color_rgb(0,0,255))
        if country_array[i,j]!=0:
            if borders[i,j]==1:
                rectarray[i,j].setFill(color_rgb(0,0,0))


for i in range(100000):
    win.update()
