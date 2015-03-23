# -*- coding: utf-8 -*-
"""
Created on Wed Jul 02 15:42:16 2014

@author: Liu
"""

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
np.random.seed(0)

x = np.random.normal(size=200)
y = np.random.normal(size=200)
v = np.sqrt(x**2+y**2)
xg, yg = np.mgrid[-2:2:100j, -2:2:100j]
vg = griddata((x, y), v, (xg, yg), method='cubic')
plt.contourf(xg, yg, vg)
plt.show()