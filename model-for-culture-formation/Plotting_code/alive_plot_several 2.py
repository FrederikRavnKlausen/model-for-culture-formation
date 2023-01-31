import numpy as np
from graphics import *
import os.path
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

#Parameters:
    #Gives the value of the fluctuations
fluc_size_initial = 0.2

    #Mountain parameters:
mountain_defence_parameter = 2 #The effect of mountains to defense is in [1,2]
mountain_perimeter_parameter = 0.6 #The effect of mountains on perimeter is [0.5,1]

    #Parameters describing the effect of sea/river borders and owning both sides of a river.
sea_border_parameter = 0.5 # Sea_border counts as 0.5
river_area_bonus = 52 # Added to area if controlling both sides of a river.

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

extended_parameter_list = [iteration_number, era, fluc_size_initial,mountain_defence_parameter,mountain_perimeter_parameter,sea_border_parameter,river_area_bonus,radius]

print(os.path.isfile(f'./Simulations2/alive_list_' + str(extended_parameter_list)+'.txt'))

#print(os.path.isfile(f"./Simulations/country_array_{(0,0,0.125,0.5,6)}.npy"))
#for i in range(20):
#    arr = np.arange(i)
#    np.save(f"array_number_{i}.npy", arr)
N=8 #number of iterations
no_timestep = 1000
firstca = np.loadtxt(f'./Simulations2/alive_list_' + str(extended_parameter_list)+'.txt')

first_list = np.flip(firstca)
first_list = np.reciprocal(first_list)
plt.plot(first_list)
plt.xlabel("timestep")
plt.ylabel("average country area")
plt.show()

for river_area_bonus in [0,8,12,16,20,24,32,40,52]:
    meta_alive_list = np.zeros(no_timestep)
    start_it = 1
    for k in range(N):
        print(k)
        extended_parameter_list = [start_it+k, era, fluc_size_initial,mountain_defence_parameter,mountain_perimeter_parameter,sea_border_parameter,river_area_bonus,radius]
        #f"./Simulations/country_array_{(k,l,0.25,0.5,6)}.npy"
        #no_conquered_array = np.load(f"./Simulations/no_conquered_array_{(k,9,0.125,0.5,6)}.npy")
        alive_list = np.loadtxt(f'./Simulations2/alive_list_' + str(extended_parameter_list)+'.txt')
        for i in range(0,no_timestep):
                meta_alive_list[i] +=  alive_list[i]

    land_area = 11000
    #Frequency conversion

    meta_alive_list =1/N*meta_alive_list

    meta_alive_list = np.flip(meta_alive_list)
    meta_alive_list =land_area*np.reciprocal(meta_alive_list)

    plt.plot(meta_alive_list, label = "$A_r$ ="+str( river_area_bonus))


plt.title("Average area of countries over time")
plt.xlabel("timestep")
plt.ylabel("average country area")
plt.legend()
plt.show()
