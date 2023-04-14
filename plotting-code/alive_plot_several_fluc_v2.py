import numpy as np
#from graphics import *
import os.path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


plt.rcParams['text.usetex'] = True

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
no_era = 100 # number of eras
length_era = 10 #length of an era
parameter_list = [length_era, fluc_size_initial,mountain_defence_parameter,mountain_perimeter_parameter,sea_border_parameter,river_area_bonus,radius]

era = 99
iteration_number = 1

extended_parameter_list = [iteration_number, era, fluc_size_initial,mountain_defence_parameter,mountain_perimeter_parameter,sea_border_parameter,river_area_bonus,radius]

filepath = './simulations_era=10_flux_varied_torus/torus_plain_200x200_alive_list_'

#print(os.path.isfile(f"./Simulations/country_array_{(0,0,0.125,0.5,6)}.npy"))
#for i in range(20):
#    arr = np.arange(i)
#    np.save(f"array_number_{i}.npy", arr)
# N=13 #number of iterations
no_timestep = 1000
# firstca = np.loadtxt(f'./sims_era=10/alive_list_' + str(extended_parameter_list)+'.txt')

# first_list = np.flip(firstca)
# first_list = np.reciprocal(first_list)
# plt.plot(first_list)
# plt.xlabel("timestep")
# plt.ylabel("average country area")
# plt.show()

# range of zoomed region
x1 = 0
x2 = 200

# select y-range for zoomed region
y1 = 0
y2 = 250

# Make the zoom-in plot:
ax = plt.axes()

ax.set_xlabel('timestep')
ax.set_ylabel('average country area')
ax.set_title('Average area of countries over time')


axins = zoomed_inset_axes(ax, 5, loc='upper left', 
    bbox_to_anchor=(-0.08,0.48,.2,.5), bbox_transform=ax.transAxes)


eval_times = [20,100,1000]
no_times = len(eval_times)
list_fluc = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
no_fluc = len(list_fluc)

meta_size_europe = np.zeros((no_fluc,no_times, N))
meta_size_torus = np.zeros((no_fluc,no_times, N))

land_area_europe = 40000 - 20787
land_area_torus = 40000


for fluc_size_initial in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    meta_alive_list = np.zeros(no_timestep)
    start_it = 1
    for k in range(N):
        print(k)
        extended_parameter_list = [start_it+k, era, fluc_size_initial,mountain_defence_parameter,mountain_perimeter_parameter,sea_border_parameter,river_area_bonus,radius]
        #f"./Simulations/country_array_{(k,l,0.25,0.5,6)}.npy"
        #no_conquered_array = np.load(f"./Simulations/no_conquered_array_{(k,9,0.125,0.5,6)}.npy")
        alive_list = np.loadtxt(filepath + str(extended_parameter_list)+'.txt')
        for i in range(0,no_timestep):
                meta_alive_list[i] +=  alive_list[i]

    land_area = 40000
    #Frequency conversion

    meta_alive_list =1/N*meta_alive_list

    meta_alive_list = np.flip(meta_alive_list)
    meta_alive_list = land_area*np.reciprocal(meta_alive_list)

    ax.plot(meta_alive_list, label = "$p$ ="+str(fluc_size_initial))

    axins.plot(meta_alive_list)
    axins.set_aspect(1.5)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    # mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")

    plt.draw() 
# plt.show()


# plt.title("Average area of countries over time")
# plt.xlabel("timestep")
# plt.ylabel("average country area")
ax.legend()
plt.show()
