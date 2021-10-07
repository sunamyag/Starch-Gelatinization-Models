# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 00:36:10 2021

@author: Sunamya Gupta

Peleg Model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from math import pi
from scipy.optimize import curve_fit

#Input parameter
fileName1 = 'chickpeat3.csv'

#Plot parameters
FS = 22
LS = 15

df1 = pd.read_csv(fileName1)
data1 = df1.to_numpy()

d = ((data1[0:,2:]/pi)**0.5)*2/2.88
time1 = data1[0:,1]

#Peleg model function
def peleg(x, p1, p2):
    D = Di + (np.multiply(x,(Df - Di))/(p1 + (np.multiply(x,p2))))
    return (D)

def rmse(predictions, target):
    return (np.sqrt(((predictions - target)**2).mean()))

x = np.linspace(100,550,450)
n = len(d[0,0:])
color=iter(cm.rainbow(np.linspace(0,1,n)))
errVal = 0

for i in range(n):
    c=next(color)
    dia = d[0:,i]
    num = len(dia)
    Df = dia[num-1]
    Di = dia[0]
    p0 = [2 ,10 ]
    popt, pcov = curve_fit(peleg, time1, dia, p0, method='dogbox')
    y = peleg(x,*popt)
    plt.plot(x,y,c=c)
    plt.plot(time1,dia,'o',c=c,markersize = 5)
    #Finding diameter at experimental time values to find RMSE
    ynew = peleg(time1,*popt)
    errVal = errVal + rmse(ynew,dia)

#calculates average RMSE over all granules
netErr = errVal/n
print("RMSE of the model is :", netErr)

plt.xlabel('Time [s]',fontsize = FS)
plt.ylabel('Diameter [$\mu$m]',fontsize = FS)
plt.tick_params(labelsize=LS)
plt.tight_layout()
plt.show
plt.savefig('Peleg.png')