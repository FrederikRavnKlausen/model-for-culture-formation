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

filepath_europe = 'C:/Users/alaurits/Desktop/Data_borders_of_Europe/simulations_era=10_flux_varied_europe/alive_list_'
filepath_torus  = 'C:/Users/alaurits/Desktop/Data_borders_of_Europe/simulations_era=10_flux_varied_torus/torus_plain_200x200_alive_list_'


eval_times = [20,100,1000]
no_times = len(eval_times)
list_fluc = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
no_fluc = len(list_fluc)

meta_size_europe = np.zeros((no_fluc,no_times, N))
meta_size_torus = np.zeros((no_fluc,no_times, N))

land_area_europe = 40000 - 20787
land_area_torus = 40000

for index_fluc in range(no_fluc):
    start_it = 1
    fluc_size_initial = list_fluc[index_fluc]
    # print(fluc_size_initial)

    for k in range(N):
        # print(k)
        extended_parameter_list = [start_it+k, era, fluc_size_initial,mountain_defence_parameter,mountain_perimeter_parameter,sea_border_parameter,river_area_bonus,radius]

        alive_list_europe = np.flip(np.loadtxt(filepath_europe + str(extended_parameter_list)+'.txt'))
        alive_list_torus = np.flip(np.loadtxt(filepath_torus + str(extended_parameter_list)+'.txt'))

        # print(alive_list_europe[950:999])

        for i in range(0,no_times):
            meta_size_europe[index_fluc, i, k] = land_area_europe * 1/alive_list_europe[eval_times[i]-1]
            meta_size_torus[index_fluc, i, k] = land_area_torus * 1/alive_list_torus[eval_times[i]-1]



mean_size_europe = np.mean(meta_size_europe, axis=2)
mean_size_torus = np.mean(meta_size_torus,axis=2)



quantiles = [0.05,0.95]


errors_europe = np.quantile(meta_size_europe, q=quantiles, axis=2)
errors_torus = np.quantile(meta_size_torus, q=quantiles, axis=2)


errors_rel_europe = errors_europe - (mean_size_europe, mean_size_europe)
errors_rel_europe = np.abs(errors_rel_europe)
print(errors_rel_europe)
# print(errors_europe[1,:,0])



for t in range(no_times):
    plt.plot(list_fluc,  mean_size_europe[:,t], '*:', label = "Europe, timestep " + str(eval_times[t]))
    plt.fill_between(list_fluc, errors_europe[0,:,t], errors_europe[1,:,t], alpha=0.2)
    # plt.errorbar(list_fluc, mean_size_europe[:,t], yerr=errors_rel_europe[:,:,t])

    plt.plot(list_fluc,  mean_size_torus[:,t], 's:', label = "Torus, timestep " + str(eval_times[t]))
    plt.fill_between(list_fluc, errors_torus[0,:,t], errors_torus[1,:,t], alpha=0.2)

# plt.plot(list_fluc, mean_size_europe[:,0])
# plt.fill_between(list_fluc, errors_europe[0,:,0], errors_europe[1,:,0], alpha=0.2)


# for k in range(N):
#     plt.plot(list_fluc, meta_size_torus[:,0,k])

plt.xlabel("Fluctuation $p$")
plt.ylabel("Average area")

plt.yscale('log')
plt.legend()
plt.show()


