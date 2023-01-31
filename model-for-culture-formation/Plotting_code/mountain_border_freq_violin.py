import numpy as np
from graphics import *
import os.path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Parameters:
    #Gives the value of the fluctuations
fluc_size_initial = 0.2



    #Parameters describing the effect of sea/river borders and owning both sides of a river.
sea_border_parameter = 0.5 # Sea_border counts as 0.5
river_area_bonus = 0 # Added to area if controlling both sides of a river.

    # Set mode_neighbours to 'circle' for a circular attacking field
    # set mode_neighbours to 'square' for a square-formed attacking field
mode_neighbours = 'circle'
radius = 4 #Radius of attack is 4.



    #Mountain parameters:
mountain_defence_parameter = 1 #The effect of mountains to defense is in [1,2]
mountain_perimeter_parameter = 0.8 #The effect of mountains on perimeter is [0.5,1]



N=20 # number of interations
start_it = 1 #start iteration
no_era = 10 # number of eras
length_era = 100 #length of an era

era = 9
iteration_number = 1

extended_parameter_list = [iteration_number, era, fluc_size_initial,mountain_defence_parameter,mountain_perimeter_parameter,sea_border_parameter,river_area_bonus,radius]
print(extended_parameter_list)

print(os.path.isfile(f'./Simulations3/country_array_' + str(extended_parameter_list)+'.npy'))
#for i in range(20):
#    arr = np.arange(i)
#    np.save(f"array_number_{i}.npy", arr)
firstca = np.load(f'./Simulations3/country_array_' + str(extended_parameter_list)+'.npy')


n = np.shape(firstca)[0]
m = np.shape(firstca)[1]
print(n)
print(m)


def find_border(country_array):
    borders = np.zeros((n,m))
    #We check whether each square is a neighbor of a different country.
    for i in range(0,n):
        for j in range(0,m):
            if 0 in np.unique(country_array[i-1:i+2, j-1:j+2]):
                if len(np.unique(country_array[i-1:i+2, j-1:j+2])) > 2:
                    borders[i,j] =1
            else:
                if len(np.unique(country_array[i-1:i+2, j-1:j+2])) > 1:
                    borders[i,j] =1
        #    if country_array[i,j]!= 0: #check land
    #            if  np.all((country_array[i-1:i+2, j-1:j+2] == country_array[i,j]) + (country_array[i-1:i+2, j-1:j+2]==0)) != True:
    #                borders[i,j] = 1
    return borders


mountain_europe_ini = np.loadtxt('mountain_background.txt')
mountain_background = np.flip(mountain_europe_ini, axis = 0)

##plots appropriate histogram
mountain_background_list = np.reshape(mountain_background, -1)
#large_numbers = np.sort(mountain_background_list)[30000:40000]
#plt.hist(large_numbers, bins="auto")
#plt.show()



#Making "time_evolution plots:"
no_era = 10
meta_borders = np.zeros((n,m,no_era))
river_border_ratios = np.zeros((10,N))
for era in range(no_era):
    print("era_no")
    print(era)
    for k in range(N):
        extended_parameter_list = [start_it+k, era, fluc_size_initial,mountain_defence_parameter,mountain_perimeter_parameter,sea_border_parameter,river_area_bonus,radius]
        cur_country_array = np.load(f'./Simulations3/country_array_' + str(extended_parameter_list)+'.npy')
        #cur_country_array = np.load(f'./Simulations/country_array_' + str(k + start_it)+str(l)+str(parameter_list)+'.npy')
        cur_border = find_border(cur_country_array)
        for i in range(0,n):
            for j in range(0,m):
                meta_borders[i,j,era] +=  cur_border[i,j]
meta_borders = 1/N*meta_borders

meta_borders_list = meta_borders[:,:,9].flatten()

mountain_background_list = mountain_background[:,:].flatten()

df = pd.DataFrame(list(zip(meta_borders_list, mountain_background_list)),
               columns =['border_freq', 'mountain_background'])




df.drop(df.index[df['mountain_background'] == 0], inplace = True)

df.drop(df.index[df['mountain_background'] + df['border_freq'] <= 0.05 ], inplace = True)

df["groupings"] = pd.cut(df.mountain_background,  6, labels=["flat", "flat","flat","hilly","mountain", "mountain high"], ordered=False)

print(df)

violin = sns.displot(df, x = "border_freq", y = "mountain_background", binwidth=(0.1, 0.1), cbar=True)
plt.show()

ax = sns.violinplot(x="groupings", y = "border_freq", data = df,cut=0)
plt.show()


def find_frequency(mountain_low, mountain_high,era):
    freq_list = []
    for i in range(n):
        for j in range(n):
            if mountain_background[i,j]>=mountain_low:
                if mountain_background[i,j]< mountain_high:
                    freq_list = np.append(freq_list,meta_borders[i,j,era])
    return freq_list

#freq_list_high = []
#era = 0
#for i in range(n):
#    for j in range(n):
#        if mountain_background[i,j]<0.8:
#            if mountain_background[i,j]>=0.6:
#                freq_list_high = np.append(freq_list_high,meta_borders[i,j,era])
freq_list_high = find_frequency(0.6, 0.8,0)
hist = np.histogram(freq_list_high, bins=[0, 0.2, 0.4, 0.6,0.8,1.0])
freq_hist = hist[0]/np.sum(hist[0])

print(hist)
print(hist[0])
print(hist[0]/np.sum(hist[0]))


plt.hist(freq_list_high, bins=[0, 0.2, 0.4, 0.6,0.8,1.0])
plt.show()

a=5
#freq_array = []
for x in range(a):
    freq_list_high = find_frequency(0.2*x, 0.2*(x+1),9)
    hist = np.histogram(freq_list_high, bins=[0, 0.2, 0.4, 0.6,0.8,1.0])
    freq_hist = hist[0]/np.sum(hist[0])
    if x ==0:
        freq_array = freq_hist
    if x>0:
        freq_array = np.column_stack((freq_array,freq_hist))



print(freq_array)
freq_array = np.flip(freq_array, axis = 0)

ax = plt.gca()
im = ax.imshow(freq_array, cmap = "inferno")
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Frequency within mountain group", rotation=-90, va="bottom")
#plt.colorbar()
ax.set_xlabel('Mountain groups')
ax.set_xticks(np.arange(5))
ax.set_yticks(np.arange(5))
ax.set_xticklabels([0.1, 0.3, 0.5, 0.7,0.9])
ax.set_yticklabels(np.flip([0.1, 0.3, 0.5, 0.7,0.9]))
ax.set_ylabel('Border frequency')
plt.show()


hist_freq_high = np.zeros(5)
for i in range(5):
    hist_freq_high[i] = len(freq_list_high[freq_list_high >= 0.2*i])
sum_high = hist_freq_high[0]
print(hist_freq_high)
for i in range(4):
    hist_freq_high[i] = (hist_freq_high[i]  - hist_freq_high[i+1])/sum_high
hist_freq_high[4] = hist_freq_high[4]/sum_high
print(hist_freq_high)
print("the following number should be 1:")
print(np.sum(hist_freq_high))



freq_list_low = []
for i in range(n):
    for j in range(n):
        if 0<mountain_background[i,j]<0.6:
            freq_list_low = np.append(freq_list_low,meta_borders[i,j,era])
plt.hist(freq_list_low, bins="auto")
plt.show()

a=5
b=5
mountain_border_array = np.zeros((a,b))
for x  in range(a):
    for y in range(b):
        for i in range(n):
            for j in range(n):
                if mountain_background[i,j]>= 0.2*x:
                    if mountain_background[i,j]< 0.2*(x+1):
                        freq_list_high = np.append(freq_list_high,meta_borders[i,j,era])

            hist_freq_high[a,i] = len(freq_list_high[freq_list_high >= 0.2*i])
        sum_high = hist_freq_high[0]
        print(hist_freq_high)
        for i in range(4):
            hist_freq_high[i] = (hist_freq_high[i]  - hist_freq_high[i+1])/sum_high
        hist_freq_high[4] = hist_freq_high[4]/sum_high
        print(hist_freq_high)
        print("the following number should be 1:")
        print(np.sum(hist_freq_high))


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
