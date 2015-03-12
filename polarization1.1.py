# -*- coding: utf-8 -*-
"""
Created on Mon Mar 02 04:18:33 2015

@author: QCSE-adm
"""

import numpy as np
import matplotlib.pyplot as plt
import libtiff
import os
from scipy.optimize import leastsq
from sub import extract

filePath='E:/QCSE data/031215 polarization/run2/'
differential_image = 1 # 1 = yes, display differential image; else = no, display mean image
deg_0 = 167

angle = []
QDint = []
for file in os.listdir(filePath):
    current_file = os.path.join(filePath, file)
    S = file.split('.')   
    mov = libtiff.TiffFile(current_file)
    mov = np.array(mov.get_tiff_array()[:,:,:], dtype='d')
    angle = np.append(angle, [int(s) for s in S if s.isdigit()])

    if np.mean(angle, dtype = 'int') == 167:

        frame = len(mov[:,0,0])
        row = len(mov[0,:,0])
        col = len(mov[0,0,:])
        scan = 5
        
        if differential_image == 1:
            mov_df = np.zeros((row, col))
            for i in range(frame-1):       
                c=mov[i,:,:]
                d=mov[i+1,:,:]
                mov_df=mov_df+np.absolute(d-c) 
            
            fig, ax = plt.subplots()
            ax.imshow(mov_df)    
     
        else: 
            mov_mean = np.mean(mov, axis=0)
            
            fig, ax = plt.subplots()
            ax.imshow(mov_mean)
            
        pts = np.array(plt.ginput(n=0, timeout=0))
        print pts
        ax.plot(pts[:,0],pts[:,1],'r+', markersize=10)
        for n in range (len(pts)):
            ax.annotate(n,xy=(pts[n,0], pts[n,1]), xytext=(pts[n,0], pts[n,1]+20),color='r')
        ax.set_xlim(0,col)
        ax.set_ylim(row,0)
        fig.canvas.draw()
        
        
        fig, ax = plt.subplots(len(pts))       
        for i in range(len(pts)):
            if differential_image == 1:  
                ax[i].imshow(mov_df[pts[i,1]-scan:pts[i,1]+scan, pts[i,0]-scan:pts[i,0]+scan])
            else:       
                ax[i].imshow(mov_mean[pts[i,1]-scan:pts[i,1]+scan, pts[i,0]-scan:pts[i,0]+scan])
   
    QDint = np.append(QDint, extract.blinkon_mean(mov, pts, scan, fig, ax))

QDint = np.reshape(QDint, [len(angle), len(pts)])

def get_color():
    for item in ['r', 'g', 'b', 'c', 'm', 'y', 'k']:
        yield item
color = get_color()
    
    
fig, ax = plt.subplots()
arti_signal = np.reshape(np.tile(np.cos(np.pi*angle/180)**2, [len(pts)]),[len(angle), len(pts)], order='F')
QDint = QDint + arti_signal*0
for i in range(len(pts)):
    acolor = next(color)    
    ax.scatter(angle-deg_0, QDint[:,i], color=acolor, marker='o')
    
    guess_min = np.min(QDint[:,i])
    guess_amplitude = np.max(QDint[:,i])-np.min(QDint[:,i])
    guess_phase = 0
    
    data_first_guess = guess_amplitude*np.cos(np.linspace(0,np.pi,1000)+guess_phase)**2 + guess_min
    
    optimize_func = lambda x: x[0]*np.cos(np.pi*angle/180+x[1])**2 + x[2] - QDint[:,i]
    est_amplitude, est_phase, est_min = leastsq(optimize_func, [guess_amplitude, guess_phase, guess_min])[0]
    
    data_fit = est_amplitude*np.cos(np.linspace(0,np.pi,1000)+est_phase)**2 + est_min
    
    ax.plot(np.linspace(0,180,1000), data_fit, acolor+ '-', label='QD{}'.format(i)+' fit')
    #ax.plot(np.linspace(0,180,1000), data_first_guess, '--', label='first guess')
    ax.set_xlim(-1,180)    
    ax.legend()
    fig.canvas.draw()