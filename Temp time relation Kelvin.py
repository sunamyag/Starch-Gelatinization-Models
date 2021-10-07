# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 18:01:58 2021

@author: Sunamya Gupta
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Plot parameters
LS = 15
FS = 22

#Input parameters
filename = 'chickpea test3 temp.csv'
tI = 140
tF = 560
tempIni = 50+273          # Enter initial and final temperatures in Kelvin
tempFinal = 90+273
timeCol = 1
tempCol = 2

df = pd.read_csv(filename)
data = df.to_numpy()
time = data[0:,timeCol]
temp = data[0:,tempCol] + 273

#Function to find temperature time variation
def logarithmic(x, x0, xf, a, b, f, g):
    xLen = len(x)
    yPred = np.empty(xLen)
    for j in range(xLen):
        if x[j] <= x0 : 
            yPred[j] = a
        elif x[j] <= xf :
            yPred[j] = f + g*np.log(x[j])
        else :
            yPred[j] = b
    return (yPred)

x = np.linspace(0,700,700)
num = len(temp)
p0 = [tI, tF, tempIni, tempFinal, 10 , 10]
popt, pcov = curve_fit(logarithmic, time, temp, p0, method='dogbox')
y = logarithmic(x,*popt)

plt.plot(x,y)
plt.plot(time,temp)
plt.xlabel('Time [s]',fontsize = FS)
plt.ylabel('Temperature [Kelvin]',fontsize = FS)
plt.tick_params(labelsize=LS)
plt.tight_layout()
plt.show
plt.savefig('Temp (Kelvin) vs time.png')