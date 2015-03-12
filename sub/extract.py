# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 06:45:52 2015

@author: QCSE-adm
"""

import numpy as np
import matplotlib.pyplot as plt

def blinkon_mean(mov, pts, scan, fig, ax):
    frame = len(mov[:,0,0])      
    timetraces1 = []
    for i in range(len(pts)):
        timetrace = np.mean(np.mean(mov[:, pts[i,1]-scan:pts[i,1]+scan, pts[i,0]-scan:pts[i,0]+scan],axis=1), axis=1)
        timetraces1 = np.append(timetraces1, timetrace)
    timetraces1 = np.reshape(timetraces1, [len(pts), frame])
        
    fig, ax = plt.subplots()
    for i in range(len(pts)):
        ax.plot(np.arange(0,frame,1), timetraces1[i,:])
        fig.canvas.draw()
        
    std = np.std(timetraces1[:,:],axis=1,ddof=1,dtype='d')
    threshold = np.mean(timetraces1, axis=1)-std/2
        
    for i in range(len(pts)):
        ax.plot(np.arange(0,frame,1), np.tile(threshold[i], frame), 'r-', label='threshold')
        fig.canvas.draw()
        
    QDintensity =[]
    for n in range(len(pts)):
        QD1 = np.mean([timetraces1[n,i] for i in range(frame) if timetraces1[n,i] > threshold[n]])
        ax.plot(np.arange(0,frame,1), np.tile(QD1, frame), 'y', label='on mean')       
        fig.canvas.draw()
        QDintensity = np.append(QDintensity, QD1)
        
    return QDintensity