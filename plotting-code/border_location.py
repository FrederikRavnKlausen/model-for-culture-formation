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
river_area_bonus = 0 # Added to area if controlling both sides of a river.

    # Set mode_neighbours to 'circle' for a circular attacking field
    # set mode_neighbours to 'square' for a square-formed attacking field
mode_neighbours = 'circle'
radius = 4 #Radius of attack is 4.
N=20 # number of interations
start_it = 1 #start iteration
no_era = 10 # number of eras
length_era = 100 #length of an era
parameter_list = [length_era, fluc_size_initial,mountain_defence_parameter,mountain_perimeter_parameter,sea_border_parameter,river_area_bonus,radius]

era = 9
iteration_number = 1

extended_parameter_list = [iteration_number, era, fluc_size_initial,mountain_defence_parameter,mountain_perimeter_parameter,sea_border_parameter,river_area_bonus,radius]

print(os.path.isfile(f'./Simulations/country_array_' + str(extended_parameter_list)+'.npy'))
#for i in range(20):
#    arr = np.arange(i)
#    np.save(f"array_number_{i}.npy", arr)
firstca = np.load(f'./Simulations/country_array_' + str(extended_parameter_list)+'.npy')

n = np.shape(firstca)[0]
m = np.shape(firstca)[1]
print(n)
print(m)


def find_border(country_array):
    borders = np.zeros((n,m))
    #We check whether each square is a neighbor of a different country.
    for i in range(0,n):
        for j in range(0,m):
            if country_array[i,j]!= 0: #check land
                if  np.all((country_array[i-1:i+2, j-1:j+2] == country_array[i,j]) + (country_array[i-1:i+2, j-1:j+2]==0)) != True:
                    borders[i,j] = 1
    return borders


meta_borders = np.zeros((n,m))
start_it = 1
for k in range(N):
    print(k)
    extended_parameter_list = [start_it+k, era, fluc_size_initial,mountain_defence_parameter,mountain_perimeter_parameter,sea_border_parameter,river_area_bonus,radius]
    cur_country_array = np.load(f'./Simulations/country_array_' + str(extended_parameter_list)+'.npy')
    #cur_country_array = np.load(f"./Simulations/country_array_{(k,9,0.15,0.5,6)}.npy")
    cur_border = find_border(cur_country_array)
    for i in range(0,n):
        for j in range(0,m):
            meta_borders[i,j] +=  cur_border[i,j]

#Frequency conversion
meta_borders = 1/N*meta_borders

meta_borders = np.rot90(meta_borders,2)

meta_borders = np.flip(meta_borders,axis =1)





plt.imshow(meta_borders)

plt.colorbar()

#for i in range(0,n):
#    for j in range(0,m):
#        if cur_country_array[i,j]==0:
#            plt.scatter(i,j)



nonland = cur_country_array == 0

nonland = np.rot90(nonland,2)

nonland = np.flip(nonland,axis =1)

cur_country_array = np.rot90(cur_country_array, 2)
cur_country_array = np.flip(cur_country_array,axis =1)

masked_data = np.ma.masked_where(cur_country_array != 0, cur_country_array)
plt.imshow(masked_data,  cmap=plt.cm.Blues)

plt.show()
plt.close()

#Making "time_evolution plots:"
no_era = 10
meta_borders = np.zeros((n,m,no_era))
for era in range(no_era):
    print("era_no")
    print(era)
    for k in range(N):
        extended_parameter_list = [start_it+k, era, fluc_size_initial,mountain_defence_parameter,mountain_perimeter_parameter,sea_border_parameter,river_area_bonus,radius]
        cur_country_array = np.load(f'./Simulations/country_array_' + str(extended_parameter_list)+'.npy')
        #cur_country_array = np.load(f'./Simulations/country_array_' + str(k + start_it)+str(l)+str(parameter_list)+'.npy')
        cur_border = find_border(cur_country_array)
        for i in range(0,n):
            for j in range(0,m):
                meta_borders[i,j,l] +=  cur_border[i,j]
    #Frequency conversion
meta_borders = 1/N*meta_borders

meta_borders = np.rot90(meta_borders,2)

meta_borders = np.flip(meta_borders,axis =1)







#fig.suptitle('Frequency of bordes', fontsize=20)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 6))

im0 = axes[0].imshow(meta_borders[:,:,1],cmap="inferno")
im1 = axes[1].imshow(meta_borders[:,:,5],cmap="inferno")
im2 = axes[2].imshow(meta_borders[:,:,9],cmap="inferno")

axes[0].set_title("Timestep = 100")
axes[1].set_title("Timestep = 500")
axes[2].set_title("Timestep = 900")

axes[0].imshow(masked_data,cmap=plt.cm.Blues)
axes[1].imshow(masked_data,cmap=plt.cm.Blues)
axes[2].imshow(masked_data,cmap=plt.cm.Blues)


axes[0].xaxis.set_visible(False)
axes[0].yaxis.set_visible(False)
axes[1].xaxis.set_visible(False)
axes[1].yaxis.set_visible(False)
axes[2].xaxis.set_visible(False)
axes[2].yaxis.set_visible(False)

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92 , 0.06, 0.04, 0.7])
fig.colorbar(im0, cax = cbar_ax)

plt.savefig("Borders_time.png")

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
