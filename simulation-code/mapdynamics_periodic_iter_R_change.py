import scipy.signal
import numpy as np
#from graphics import *
import time
import math



# Set plot to 1 og 0 depending whether to plot or not.
plot = 0


    #Mountain parameters:
mountain_defence_parameter = 2 #The effect of mountains to defense is in [1,2]
mountain_perimeter_parameter = 0.5 #The effect of mountains on perimeter is [0.5,1]

    #Parameters describing the effect of sea/river borders and owning both sides of a river.
sea_border_parameter = 0.5 # Sea_border counts as 0.5
river_area_bonus = 8 # Added to area if controlling both sides of a river.

    # Set mode_neighbours to 'circle' for a circular attacking field
    # set mode_neighbours to 'square' for a square-formed attacking field
mode_neighbours = 'circle'
radius = 8 #Radius of attack is 4.

##Looping over many iterations
N=1 # number of interations
start_it = 1 #start iteration
no_era = 10 # number of eras
length_era = 100 #length of an era

    # Sets the desired zoom-scale of the map
    #zoom_scale = 40

    #string_input_map = 'map_zoom' + str(zoom_scale) + '.txt'
    #string_input_curvature = 'curvature_zoom' + str(zoom_scale) + '.txt'

map_europe_ini = np.loadtxt('map200x200_v3.txt')

map_europe = np.flip(map_europe_ini, axis = 0)


#curvature_europe_ini = np.loadtxt('curvature200x200.txt')
# curvature_europe = np.flip(curvature_europe_ini, axis = 0)
mountain_europe_ini = np.loadtxt('mountain_background.txt')

mountain_background = np.flip(mountain_europe_ini, axis = 0)

rivers_ini = np.genfromtxt('rivers200x200-curated7.csv', delimiter=';')
rivers_ini = np.reshape(rivers_ini, (200,200))
rivers = np.flip(rivers_ini, axis = 0)

    # 2nd axis
n = map_europe_ini.shape[0]

    # 1st axis
m = map_europe_ini.shape[1]

#Parameters:
    #Gives the value of the fluctuations
fluc_size_initial = 0.2
    # Gives the fluctuation of strengths
def fluctuations(random_number_counter, fluc_size):
    return 1 + 2 * fluc_size * (random_numbers[random_number_counter] - 1/2)

    # Allows for time-varied fluctuations
def fluctuation_size_function(time_step):
    return fluc_size_initial


random_number_length = 1000000
def new_random_numbers():
    #print("making new array of random numbers")
    return np.random.random_sample(random_number_length,)


random_number_counter = 0
random_numbers = new_random_numbers()


# Defines the global strength function of the countries
def str_fun(perimeter, area):
	return np.sqrt(area) / perimeter
    #(1 - math.exp(- area / 5)) * np.sqrt(area) / perimeter
    #


# Changes how perimeter is calculated per square
def perim_fun(no_neighbors):
    return 1

# Calculate neighbours for attacking a square at coorinates (i, j)
if mode_neighbours == 'circle':
    def neighbours_of_square(i, j, radius, array):

        min_1 = max(0, i - radius)
        max_1 = min(n, i + radius + 1)

        min_2 = max(0, j - radius)
        max_2 = min(m, j + radius + 1)

        indices_1 = range(min_1, max_1)
        indices_2 = range(min_2, max_2)

        neighbors = np.take(np.take(array, indices_1, axis =0), indices_2, axis=1)

        for index_1 in indices_1:
            for index_2 in indices_2:
                dist_to_center_sq = (index_1 - i)**2 + (index_2 - j)**2

                # If distance of square to center (i, j) is too big, classify square as 'non-land'
                if dist_to_center_sq > radius**2:

                    index1 = index_1 - min_1
                    index2 = index_2 - min_2

                    neighbors[index1, index2] = non_land

        return neighbors


if mode_neighbours == 'square':
    def neighbours_of_square(i, j, radius):

        indices_1 = []
        indices_2 = []

        for shift in range(-radius, radius + 1):

            indices_1 += [np.mod(i + shift, n)]
            indices_2 += [np.mod(j + shift, m)]

        neighbors = np.take(np.take(country_array, indices_1, axis =0), indices_2, axis=1)

        return neighbors



#Defines the 3x3 neighbourhood to check if square is on perimeter
def neighbours_3x3(i,j, array):

    min_1 = max(0, i-1)
    max_1 = min(i+2, n)

    min_2 = max(0, j-1)
    max_2 = min(j+2, m)

    indices_1 = range(min_1, max_1)
    indices_2 = range(min_2, max_2)

    neighbours = np.take(np.take(array, indices_1, axis =0), indices_2, axis=1)

    return neighbours



# Define land-areas
land = np.ones((n,m))
non_land = 0
non_land_area = 0


    # Modify how each square contributes to the perimeter
perimeter_background = np.zeros((n,m))
for i in range(0,n):
    for j in range(0,m):
    	perimeter_background[i,j] = 1+(mountain_perimeter_parameter-1)*(mountain_background[i, j])


    #Modify how each square gives a defensive bonus
defence_background = np.zeros((n,m))
for i in range(0,n):
    for j in range(0,m):
    	defence_background[i,j] = 1+(mountain_defence_parameter-1)*(mountain_background[i, j])


    # Check whether the squares are part of the land or the sea
for i in range(0, n):
    for j in range(0, m):
        if map_europe[i, j] == 0 or rivers[i, j] == 1:
            land[i, j] = 0

non_land_area = sum(sum(land == 0))

land_area = sum(sum(land == 1))


parameter_list = [length_era, fluc_size_initial,mountain_defence_parameter,mountain_perimeter_parameter,sea_border_parameter,river_area_bonus,radius]


for iteration_number in range(start_it, N+start_it):
    np.random.seed()
    print("We are now at iteration: ", iteration_number)
    #print(iteration_number)

    if plot == 1:
        windowsize = 600

        win = GraphWin("Borders of Europe", windowsize * m / n, windowsize, autoflush=False)
        win.setCoords(0,0,m,n)



        # Define an array of rectangles to be drawn
        rectarray = []
        for i in range(0,n):
            for j in range(0,m):
                tmprect = Rectangle(Point(j,i),Point(j+1,i+1))
                tmprect.setOutline(color_rgb(140,140,140))
                tmprect.draw(win)
                rectarray.append(tmprect)
        rectarray = np.reshape(rectarray, (n,m))



    # country # 0 is the non-land, then there are n * n lands
    no_alive = n * m + 1 - non_land_area
    countries_alive = np.array(range(no_alive))



    # Initialize countries
    country_array = np.zeros((n, m), dtype = int)
    k = 1
    for i in range(0,n):
        for j in range(0,m):
            if land[i, j] == 1:
                country_array[i, j] = k
                k += 1


    # Initialize variable counting how often a square has been conquered
    no_conquered_array = np.zeros((n, m), dtype = int)


    #country_array = np.array(range(n*n)).reshape(n, n)


    non_land_color = np.array([[255, 255, 255]])


    country_colors = np.concatenate(
    	(non_land_color, np.random.randint(256, size=(no_alive - 1, 3)) ))





    # Calculate the area and perimeters of the countries
    country_perimeters = np.zeros(no_alive, dtype=int)
    country_areas = np.zeros(no_alive, dtype=int)
    alive_list = []
    no_conquered_squares_list =[]

    for i in range(0,n):
        for j in range(0,m):
            if land[i ,j] == 1:
                cur_country = country_array[i, j]
                cur_country_index = np.where(countries_alive == cur_country)[0][0]
                country_areas[cur_country_index] += 1

                neighbors = neighbours_3x3(i,j, country_array)

                #indices_1 = [np.mod(i-1,n), np.mod(i,n), np.mod(i+1,n)]
                #indices_2 = [np.mod(j-1,m), np.mod(j,m), np.mod(j+1,m)]
                #neighbors = np.take(np.take(country_array, indices_1, axis = 0), indices_2, axis=1)

                if( np.count_nonzero(neighbors != cur_country) != np.count_nonzero(neighbors == non_land) ):
                    no_neighbors = np.count_nonzero(neighbors != cur_country) - np.count_nonzero(neighbors == non_land)
                    country_perimeters[cur_country_index] += perimeter_background[i,j] * perim_fun(no_neighbors)

                elif np.count_nonzero(neighbors != cur_country) != 0:
                    no_neighbors = np.count_nonzero(neighbors != cur_country) - np.count_nonzero(neighbors == non_land)
                    country_perimeters[cur_country_index] += perimeter_background[i,j] * perim_fun(no_neighbors)* sea_border_parameter

                elif neighbors.size != 9:
                	country_perimeters[cur_country_index] += perimeter_background[i,j]

    clock = time.time()

    for era in range(0,no_era):

        #p.save(f"./Simulations/no_conquered_array_{iteration_number, era}.npy", no_conquered_array)
        #np.save(f"./Simulations/country_array_{iteration_number, era}.npy", country_array )

        for timer_step in range(0,length_era):

            time_step = era*length_era + timer_step


            #Update list of alive countries
            countries_alive = np.concatenate(( np.zeros(1), countries_alive[country_areas != 0]))
            no_alive = len(countries_alive)


            if plot ==1:
                # Color the rectangles
                for i in range(0,n):
                    for j in range(0,m):
                        cur_country = country_array[i,j]
                        cur_color = country_colors[cur_country,:]
                        rectarray[i,j].setFill(color_rgb(cur_color[0],cur_color[1],cur_color[2]))

                win.update()



            # Set fluctuations
            fluc_size = fluctuation_size_function(time_step)



            # Calculate the area and perimeters of the countries
            country_perimeters = np.zeros(no_alive)
            country_areas = np.zeros(no_alive, dtype=int)

            for i in range(0,n):
                for j in range(0,m):
                    if land[i, j] == 1:

                        cur_country = country_array[i,j]
                        cur_country_index = np.where(countries_alive == cur_country)[0][0]
                        country_areas[cur_country_index] += 1

                        neighbors = neighbours_3x3(i, j, country_array)

                        #indices_1 = [np.mod(i-1,n), np.mod(i,n), np.mod(i+1,n)]
                        #indices_2 = [np.mod(j-1,m), np.mod(j,m), np.mod(j+1,m)]
                        #neighbors = np.take(np.take(country_array, indices_1, axis = 0), indices_2, axis=1)


                        if( np.count_nonzero(neighbors != cur_country) != np.count_nonzero(neighbors == non_land) ):
                            no_neighbors = np.count_nonzero(neighbors != cur_country) - np.count_nonzero(neighbors == non_land)
                            country_perimeters[cur_country_index] += perimeter_background[i,j] * perim_fun(no_neighbors)

                        elif np.count_nonzero(neighbors != cur_country) != 0:
                            no_neighbors = np.count_nonzero(neighbors != cur_country) - np.count_nonzero(neighbors == non_land)
                            country_perimeters[cur_country_index] += perimeter_background[i,j] * perim_fun(no_neighbors)* sea_border_parameter

                        elif neighbors.size != 9:
                        	country_perimeters[cur_country_index] += perimeter_background[i,j]

        # calculates effects of rivers:
            for i in range(0,n):
                for j in range(0,m):
                    if rivers[i,j]== 1:
                    	#Find 3x3 array of closest neighbours
                        around_array = country_array[i-1: i+2 ,j-1: j+2] #okay since all rivers are inside land
                        non_river_neighbors = 9.0 - np.count_nonzero(rivers[i-1:i+2 ,j-1:j+2])
                        if np.size(np.unique(around_array)) == 2: ## bonus for each country that surrounds a river square
                        	# Finds index of the bordering country
                        	cur_country = np.unique(around_array)[1]
                        	cur_country_index = np.where(countries_alive == cur_country)[0][0]

                        	#Increases area by river_bonus
                        	country_areas[cur_country_index] += river_area_bonus
                        	for l in range(i-1,i+2):  ## find country strength and add compensation to each
	                            for h in range(j-1, j+2):
	                                if land[l,h] == 1:
	                                    country_perimeters[cur_country_index]-= sea_border_parameter * 2/non_river_neighbors
	                                    ##In this parametrization rivers borders count "sea_border_parameter" on average



            # Calculate the global "strength" of a country
            country_strengths = np.zeros(no_alive)
            for k in range(0, no_alive):
                if(country_perimeters[k]):
                    country_strengths[k] = str_fun(country_perimeters[k], country_areas[k])

            country_strengths[non_land] = 0




            # Calculate the local "strength points" of every cell:
            #strength_points = np.zeros((n, m, no_alive))
            winner_array = np.zeros((n,m), dtype = int)

            for i in range(0,n):
                for j in range(0,m):
                    if land[i, j] == 1:

                        neighbors = neighbours_of_square(i, j, radius, country_array)

                        unique_neighbors = np.unique(neighbors)

                        strength_points_i_j = np.zeros(no_alive)
                        for k in unique_neighbors:

                            no_country = sum(sum(neighbors == k))

                            index_country = np.where(countries_alive == k)[0][0]

                            strength_points_i_j[index_country] = country_strengths[index_country] * no_country * fluctuations(random_number_counter, fluc_size)
                            random_number_counter = random_number_counter + 1
                            if random_number_counter == random_number_length-2:
                                random_numbers = new_random_numbers()
                                print("Generated new random numbers")
                                random_number_counter = 0
                            #strength_points[i, j, index_country] = country_strengths[index_country] * no_country * fluctuations(1, fluc_size)

                            if countries_alive[index_country] == country_array[i, j]:

                                strength_points_i_j[index_country] = strength_points_i_j[index_country] * defence_background[i, j]


                            #    strength_points[i, j, index_country] = strength_points[i, j, index_country] * defence_background[i, j]

                        # Determines the victor
                        winner_index_i_j = np.argmax(strength_points_i_j)
                        winner_array[i, j] = countries_alive[winner_index_i_j]



            # Measure if winner is the same as previous land owner
            no_conquered_array += winner_array != country_array


            ratio_of_conquered_squares = sum(sum(winner_array != country_array)) / land_area
            no_conquered_squares = sum(sum(winner_array != country_array))


            # Update according to the winner:
            country_array = winner_array




            clock_change = time.time() - clock
            clock = time.time()

            alive_list = np.append(no_alive, alive_list)
            no_conquered_squares_list = np.append(no_conquered_squares, no_conquered_squares_list)

            #print("Countries alive: ", no_alive)
            #print("Timestep: ", time_step)
            #print("Time for last timestep", clock_change)
            #print("Highest ratio of strengths: ", max(country_strengths) / min(country_strengths[country_strengths != 0]))
            #print("Number of squares conquered last timestep: ", no_conquered_squares)
            #print("Most conquered square has been conquered ", np.amax(no_conquered_array), " times")

            print("iteration: ", iteration_number, "era: ", era, "timestep: ", time_step, "time: ", clock_change)

            extended_parameter_list = [iteration_number, era, fluc_size_initial,mountain_defence_parameter,mountain_perimeter_parameter,sea_border_parameter,river_area_bonus,radius]

            filename_no_conquered = './Simulations/no_conquered_array_' + str(extended_parameter_list)+'.npy'
            filename_country_array = './Simulations/country_array_' + str(extended_parameter_list)+'.npy'
            filename_alive_list = './Simulations/alive_list_' + str(extended_parameter_list)+'.txt'
            filename_no_conquered_squares_list = './Simulations/no_conquered_squares_list_' + str(extended_parameter_list)+'.txt'

        np.save(filename_no_conquered, no_conquered_array)
        np.save(filename_country_array, country_array)
        np.savetxt(filename_alive_list, alive_list)
        np.savetxt(filename_no_conquered_squares_list, no_conquered_squares_list)
## Saving country_array:
