# -*- coding: utf-8 -*-
"""
Code to plot the results of the data obtained from csv file in python.
-Sunamya Gupta
"""

import pandas as pd 
import matplotlib.pyplot as plt
import itertools as itl

#Plot parameters
FS = 25 #font size
LS = 15 #label size
FW = 16 #figure width
FH = 9  #figure height
MS = 30 #marker size for diameter
TL = 6  #tick length
TW = 2  #tick width
markers = ['o','v','2','x','+','*','4','^','<','s','3', '>', '1','P']
colors = ['b','g','r','m','c','k']

#Input parameters used in the code
numRows = 5 #Number of empty rows in the csv file before the data begins
df = pd.DataFrame(pd.read_csv(r'C:\Users\user\Desktop\MITACS\Project Work\Yellowpea test4-20210601T152857Z-001\Yellowpea test4\data.csv', sep = ',', skiprows = numRows))

df = df.drop('temperature (C)', axis = 1)
iniTime = df.at[0,'time (units)']
df['time (units)'] = (df['time (units)'] - iniTime)/60000

#Finding the diameter of the granule from the data obtained.
df.iloc[:,1:] = (((df.iloc[:,1:])*(7/22))**(1/2))*(2/2.88)

#Plotting the data
mcList = list(itl.product(markers, colors))
y = df.columns[1:]
plt.rcParams["figure.figsize"] = (FW,FH)
k = 1
for col in y:
    mrkr,clr = mcList[k]
    plt.scatter(df['time (units)'], df[col], s = MS, marker = mrkr, color = clr, label = k)
    k = k + 1

plt.xlabel('Time (minutes)',fontsize = FS)
plt.ylabel('Diameter (\u03bcm)',fontsize = FS)
plt.tick_params(axis = 'both', direction = 'out', length = TL, width = TW, labelsize = LS)
plt.tight_layout()
plt.savefig('ResultsImage.png')
