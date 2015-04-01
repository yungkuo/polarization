# -*- coding: utf-8 -*-
"""
Created on Mon Mar 02 04:18:33 2015

@author: QCSE-adm
"""

import numpy as np
import matplotlib.pyplot as plt
import libtiff
import os
#from scipy.optimize import leastsq
from scipy.optimize import curve_fit
from sub import extract


filePath='E:/QCSE data/031215 polarization/run4/'
differential_image = 1 # 1 = yes, display differential image; else = no, display mean image
deg_0 = 167
scan = 3

angle = []
QDint = []
for file in os.listdir(filePath):
    current_file = os.path.join(filePath, file)
    S = file.split('.')   
    mov = libtiff.TiffFile(current_file)
    mov = np.array(mov.get_tiff_array()[:,:,:], dtype='d')
    angle = np.append(angle, [int(s) for s in S if s.isdigit()])

    if np.mean(angle, dtype = 'int') == deg_0:

        frame = len(mov[:,0,0])
        row = len(mov[0,:,0])
        col = len(mov[0,0,:])
        
        
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
            
        #pts = np.array(plt.ginput(n=0, timeout=0))
        pts = np.array([[ 108.42546584,  212.76708075],
                        [ 109.41304348,  472.52732919],

                        [ 110.4068323,   206.79813665],
                        [ 111.40062112,  466.00080745],

                        [ 175.00310559,  132.27639752],
                        [ 184.94720497,  386.68012422],


                        [ 183.45341615,  148.17080745],
                        [ 192.90372671,  403.58074534],
  

               ])

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
    for item in ['b', 'g', 'r', 'c', 'm', 'y', 'k', '0.9', '0.6', '0.3']:
        yield item
color = get_color()

def cos_sqr(x, *p):
    A, phi, b = p
    return A*np.cos(x+phi)**2 + b

    
fig, ax = plt.subplots(len(pts))
arti_signal = np.reshape(np.tile(np.cos(np.pi*(angle-deg_0)*2/180)**2, [len(pts)]),[len(angle), len(pts)], order='F')
QDint = QDint + arti_signal*0
for i in range(len(pts)):
    acolor = next(color)    
    ax[i].scatter((angle-deg_0)*2, QDint[:,i], color=acolor, marker='o')
    
    guess_min = np.min(QDint[:,i])
    guess_amplitude = np.max(QDint[:,i])-np.min(QDint[:,i])
    guess_phase = (angle[int(np.where(QDint==QDint[:,i].max())[0])]-deg_0)*2*np.pi/180
    guess = [guess_amplitude, guess_phase, guess_min]
    data_first_guess = cos_sqr(np.linspace(0,np.pi*2,1000), *guess)
    guess_amplitude*np.cos(np.linspace(0,np.pi,1000)+guess_phase)**2 + guess_min
    
    fit_prmt, pcov = curve_fit(cos_sqr, (angle-deg_0)*2*np.pi/180, QDint[:,i], p0=guess)    
    data_fit = cos_sqr(np.linspace(0,np.pi*2,1000), *fit_prmt)
    perr = np.sqrt(np.diag(pcov))
    
    #optimize_func = lambda x: x[0]*np.cos(np.pi*(angle-deg_0)*2/180+x[1])**2 + x[2] - QDint[:,i]
    #est_amplitude, est_phase, est_min = leastsq(optimize_func, [guess_amplitude, guess_phase, guess_min])[0]
    
    #data_fit = est_amplitude*np.cos(np.linspace(0,np.pi,1000)+est_phase)**2 + est_min
    
    ax[i].plot(np.linspace(0,360,1000), data_fit, color=acolor, linestyle='-', label='QD{}'.format(i)+' fit')
    #ax.plot(np.linspace(0,180,1000), data_first_guess, '--', label='first guess')
    ax[i].set_xlim((angle.min()-deg_0)*2-1,(angle.max()-deg_0)*2+1)    
    ax[i].legend()
    fig.canvas.draw()