# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 13:53:41 2021

@author: Kristof
"""

import dataLib as DL
import videoLib as VL
import nozzleLibExpNum as NL
import numpy as np
import matplotlib.pyplot as plt


names = DL.getFiles("Nozzle 25_11_21 K/", "ALL_data.txt")

for filename in range(0, len(names), 1 ):

    filename = names[filename]
    
    data = DL.import2DDataFromTxt(filename, skip = 1, transpose = False)
    
    d = []
    l = []
    
    for i in range(0, len(data), 1):
        d.append(data[i][0])
        l.append(data[i][1])
    
    
    plt.figure()
    plt.plot(range(0, len(d), 1), d, "-k")
    
    if len(d)<8000: plt.xlim(0, 2000) 
    else: plt.xlim(0, 8000) 
    
    plt.ylim(0.08, 0.105)
    
    plt.grid()
    plt.title(filename[18:-13])
    plt.savefig(str(filename[0:-13])+"_D.png")
    
    plt.figure()
    plt.plot(range(0, len(l), 1), l, "-k")
    
    if len(l)<8000: plt.xlim(0, 2000) 
    else: plt.xlim(0, 8000) 
    
    plt.ylim(0.8, 2.55)
    
    plt.grid()
    plt.title(filename[18:-13])
    plt.savefig(str(filename[0:-13])+"_L.png")