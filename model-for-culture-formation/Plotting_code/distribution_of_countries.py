import numpy as np
#from graphics import *
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
N=1 # number of interations
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

from collections import Counter
dictonary = Counter(firstca.flatten())

print(dictonary)
dictonary.pop(0)
country_sizes = dictonary.values()


data = country_sizes
print("Hallo")
print(np.sum(list(data)))
print(min(data))
print(max(data))
bins = np.arange(0, 500, 5) # fixed bin size

plt.xlim([min(data), 500])

plt.hist(data, bins=bins, alpha=0.5)
plt.title('Histogram of sizes of countries')
plt.xlabel('country size')
plt.ylabel('count')

plt.show()
plt.close()

list_of_European = [3968, 603, 552, 499, 450, 385,357,338,242,238,207,148,131,111,102,93,88,83,78,70,65,57,51,49,45,44,41,33,29,26,783,20,14,11,7,3,3,1,1,1,1,1,10,30]

data = list_of_European
print(min(data))
print(max(data))
bins = np.arange(0, 500, 5) # fixed bin size

plt.xlim([min(data), 500])

plt.hist(data, bins=bins, alpha=0.5)
plt.title('Histogram of sizes of countries')
plt.xlabel('country size')
plt.ylabel('count')

plt.show()
