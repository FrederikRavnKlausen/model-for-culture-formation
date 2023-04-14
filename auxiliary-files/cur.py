import numpy as np
from graphics import *
import os.path
import matplotlib.pyplot as plt
import math

def perimeter_background_function(curvature):
    return 2 / (1 + np.exp(curvature / 50))


curvature_europe = np.loadtxt('curvature200x200.txt')



plt.imshow(perimeter_background_function(curvature_europe), vmin = 0.3, vmax = 0.8)

plt.colorbar()

plt.show()

plt.close()
