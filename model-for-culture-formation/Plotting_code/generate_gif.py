import numpy as np
from array2gif import write_gif
from graphics import *
import matplotlib.pyplot as plt
import time

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
no_era = 1000 # number of eras
length_era = 1 #length of an era
parameter_list = [length_era, fluc_size_initial, mountain_defence_parameter, mountain_perimeter_parameter, sea_border_parameter, river_area_bonus, radius]


iteration_number = 1


mode = 'torus'
# Set mode to 'europe' for the map of europe 
# Set mode to 'torus' for the torus
# If mode = 'torus' change sizes below
# 
# 

size1 = 200
size2 = 200

if mode == 'torus':
	size1 = 80
	size2 = 80








clock = time.time()



data_load = np.zeros((no_era,size2,size1))


if mode == 'europe':
	string_load = './Simulations/country_array_'

elif mode == 'torus':
	string_load = './Simulations/torus_plain_' + str(size2) + 'x' + str(size1) + '_country_array_'



no_countries = 1000
t=0
clock = time.time()

while no_countries > 254:
	extended_parameter_list = [iteration_number, t, fluc_size_initial,mountain_defence_parameter,mountain_perimeter_parameter,sea_border_parameter,river_area_bonus,radius]
	filename_country_array = string_load + str(extended_parameter_list)+'.npy'	
	ini = np.load(filename_country_array)
	no_countries = len(np.unique(ini))
	clock_change = time.time() - clock
	clock = time.time()
	# print(t, 'too many countries', 'time:', clock_change)
	t += 1

time_initial = t - 1
print(time_initial)

list_end_countries = np.unique(ini)
# print(list_end_countries, no_countries)

# Generate colours

non_land_color = np.array([[255, 255, 255]])

country_colors = np.concatenate(
    	(non_land_color, np.random.randint(255, size=(256,3))))





for t in range(no_era):
	clock_change = time.time() - clock
	clock = time.time()
	era = t
	print(era, 'loading', 'time:', clock_change)
	extended_parameter_list = [iteration_number, era, fluc_size_initial,mountain_defence_parameter,mountain_perimeter_parameter,sea_border_parameter,river_area_bonus,radius]
	filename_country_array = string_load + str(extended_parameter_list)+'.npy'
	data_load[t,:,:] = np.load(filename_country_array)


data_load = data_load.astype('int32')
data_load = np.flip(data_load, axis=2)
data_load = np.flip(data_load, axis=1)
data_load = np.flip(data_load, axis=2)


# print(data_load)





dataset = []


#Generate colour map at each timestep

for t in range(no_era):
	clock_change = time.time() - clock
	clock = time.time()
	era = t
	print(era, 'colouring', 'time:', clock_change)

	col_map = np.zeros((3,size2,size1))
	for i in range(size2):
		for j in range(size1):
			index = np.random.randint(255) + 1
			for col in range(3): #red/green/blue
				cur_country = data_load[t,i,j]
				# print(cur_country)
				if t < time_initial + 2:	
					if cur_country in list_end_countries:
						cur_country_index = np.where(list_end_countries == cur_country)[0][0]
						# print(t, i, j, cur_country, cur_country_index)
						col_map[col,i,j] = country_colors[cur_country_index,col]

					else:
						col_map[col,i,j] = country_colors[index,col]
				else:
					cur_country_index = np.where(list_end_countries == cur_country)[0][0]
					# print(t, i, j, cur_country, cur_country_index)
					col_map[col,i,j] = country_colors[cur_country_index,col]

	dataset = dataset + [col_map]

# print(dataset)





# dataset = [
#     np.array([
#         [[255, 0, 0], [255, 0, 0]],  # red intensities
#         [[0, 255, 0], [0, 255, 0]],  # green intensities
#         [[0, 0, 255], [0, 0, 255]]   # blue intensities
#     ]),
#     np.array([
#         [[0, 0, 255], [0, 0, 255]],
#         [[0, 255, 0], [0, 255, 0]],
#         [[255, 0, 0], [255, 0, 0]]
#     ])
# ]


if mode == 'europe':
	filename_out = './gifs/europe_' + str(parameter_list) + '.gif'

elif mode == 'torus':
	parameter_list = [length_era, fluc_size_initial, radius]
	filename_out = './gifs/torus_' + str(size1) + 'x' + str(size2) + '_' + str(parameter_list) + '.gif'

clock = time.time()

print('generating file')
write_gif(dataset, filename_out, fps=5)

clock_change = time.time() - clock
print('generated file', 'time', clock_change)


# # or for just a still GIF
# write_gif(dataset[0], 'rgb.gif')