import numpy as np
#from graphics import *
import os.path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


plt.rcParams['text.usetex'] = True
plt.rc('font', size=15)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


#Parameters:
    #Gives the value of the fluctuations
fluc_size_initial = 0.2

    #Mountain parameters:
mountain_defence_parameter = 2 #The effect of mountains to defense is in [1,2]
mountain_perimeter_parameter = 0.6 #The effect of mountains on perimeter is [0.5,1]

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

extended_parameter_list = [iteration_number, era, fluc_size_initial,mountain_defence_parameter,mountain_perimeter_parameter,sea_border_parameter,river_area_bonus,radius]

filepath = 'C:/Users/alaurits/Desktop/temp/simulations_era=100/alive_list_'

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
x2 = 20

# select y-range for zoomed region
y1 = 0
y2 = 50

aspect=0.4

# Make the zoom-in plot:
ax = plt.axes()

ax.set_xlabel('Timestep')
ax.set_ylabel('Average area')
# ax.set_title('Europe')


# axins = zoomed_inset_axes(ax, 20, loc='upper left', 
#     bbox_to_anchor=(0.4,1.78,.2,.5), bbox_transform=ax.transAxes)


# eval_times = [20,100,1000]
# no_times = len(eval_times)
list_area = [0,8,12,16,20,24,32,40,52]
no_area = len(list_area)

meta_size = np.zeros((no_area,no_timestep, N))

land_area = 40000 - 20787


for index_area in range(no_area):
    start_it = 1
    river_area_bonus = list_area[index_area]
    for k in range(N):
        print(k)
        extended_parameter_list = [start_it+k, era, fluc_size_initial,mountain_defence_parameter,mountain_perimeter_parameter,sea_border_parameter,river_area_bonus,radius]
        #f"./Simulations/country_array_{(k,l,0.25,0.5,6)}.npy"
        #no_conquered_array = np.load(f"./Simulations/no_conquered_array_{(k,9,0.125,0.5,6)}.npy")
        alive_list = np.flip(np.loadtxt(filepath + str(extended_parameter_list)+'.txt'))
        for i in range(0,no_timestep):
            meta_size[index_area, i, k] = land_area * 1/alive_list[i]


mean_size = np.mean(meta_size, axis=2)

quantiles = [0.05,0.95]

errors = np.quantile(meta_size, q=quantiles, axis=2)
print(mean_size.shape)
print(errors.shape)



for index_area in range(no_area):
    river_area_bonus = list_area[index_area]

    ax.plot(list(range(1,no_timestep+1)),mean_size[index_area,:], label = "$A_r$ ="+str(river_area_bonus))
    ax.fill_between(list(range(1,no_timestep+1)), errors[0,index_area, :], errors[1,index_area, :], alpha=0.2)

    # axins.plot(list(range(no_timestep)),mean_size[index_area,:])
    # axins.fill_between(list(range(no_timestep)), errors[0,index_area, :], errors[1,index_area, :], alpha=0.2)
    # axins.set_aspect(aspect)
    # axins.set_xlim(x1, x2)
    # axins.set_ylim(y1, y2)

    # mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")

    plt.draw() 
# plt.show()


# plt.title("Average area of countries over time")
# plt.xlabel("timestep")
# plt.ylabel("average country area")
ax.legend()
plt.show()
