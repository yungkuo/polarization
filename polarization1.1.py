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
#from scipy.optimize import curve_fit
from sub import extract

filePath = 'E:/QCSE data/031215 polarization/run4/'
#filePath = '/Users/yung/run4/'
differential_image = 1 # 1 = yes, display differential image; else = no, display mean image
deg_0 = 167
scan = 3
displacement = -4
n1 = 1.5
n2 = 1.000


angle = []
QDint = []
QDstd = []
offint = []
offstd = []
BGint = []
BGstd = []
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
                #ax[i,1].imshow(mov_df[pts[i,1]-scan+displacement:pts[i,1]+scan+displacement, pts[i,0]-scan:pts[i,0]+scan])
            
            else:       
                ax[i].imshow(mov_mean[pts[i,1]-scan:pts[i,1]+scan, pts[i,0]-scan:pts[i,0]+scan])
                #ax[i,1].imshow(mov_mean[pts[i,1]-scan+displacement:pts[i,1]+scan+displacement, pts[i,0]-scan:pts[i,0]+scan])

  
    blinkon_int, blinkon_std = extract.blinkon_mean(mov, pts, scan, fig, ax) 
    blinkoff_int, blinkoff_std = extract.blinkoff_mean(mov, pts, scan, fig, ax)
    #BG_int, BG_std = extract.bg_mean(mov, pts, scan, displacement, fig, ax)
    QDint = np.append(QDint, blinkon_int)
    QDstd = np.append(QDstd, blinkon_std)
    offint = np.append(offint, blinkoff_int)
    offstd = np.append(offstd, blinkoff_std)
    #BGint = np.append(BGint, BG_int)
    #BGstd = np.append(BGstd, BG_std)


QDint = np.reshape(QDint, [len(angle), len(pts)])
QDstd = np.reshape(QDstd, [len(angle), len(pts)])
offint = np.reshape(offint, [len(angle), len(pts)])
offstd = np.reshape(offstd, [len(angle), len(pts)])
#BGint = np.reshape(BGint, [len(angle), len(pts)])
#BGstd = np.reshape(BGstd, [len(angle), len(pts)])

#for i in range (len(pts)):
#    fig, ax = plt.subplots()
#    timetraces1 = np.mean(np.mean(mov[:, pts[i,1]-scan:pts[i,1]+scan, pts[i,0]-scan:pts[i,0]+scan],axis=1), axis=1)
#    ax.plot(np.arange(0,frame,1), timetraces1)
#    ax.plot(np.arange(0,frame,1),np.tile(QDint[26,i],frame),label='on')
#    ax.plot(np.arange(0,frame,1),np.tile(QDstd[26,i]+QDint[26,i],frame),label='on std')
#    ax.plot(np.arange(0,frame,1),np.tile(offint[26,i],frame), label='off')
#    ax.plot(np.arange(0,frame,1),np.tile(offstd[26,i]+offint[26,i],frame), label='off std')
#    ax.legend()

#QDint = QDint-offint


def get_color():
    for item in ['b', 'g', 'r', 'c', 'm', 'y', 'k', '0.3', '0.6', '0.9']:
        yield item
        
color = get_color()

def cos_sqr(x, *p):
    A, phi, b = p
    return A*np.cos(phi-x)**2 + b

A = []
phi = []
b = []

    
fig, ax = plt.subplots(len(pts))
arti_signal = np.reshape(np.tile(np.cos(np.pi*(angle-deg_0)*2/180)**2, [len(pts)]),[len(angle), len(pts)], order='F')
QDint = QDint + arti_signal*0
for i in range(len(pts)):
    acolor = next(color)    
    ax[i].scatter((angle-deg_0)*2, QDint[:,i], color=acolor, marker='o')
    ax[i].errorbar((angle-deg_0)*2, QDint[:,i], xerr=None, yerr=QDstd[:,i], ecolor=acolor, fmt='none')    
    #ax[i,1].scatter((angle-deg_0)*2, BGint[:,i], color=acolor, marker='^')
    #ax[i,1].errorbar((angle-deg_0)*2, BGint[:,i], xerr=None, yerr=BGstd[:,i], ecolor=acolor, fmt='none')     
    
    
    guess_min = np.min(QDint[:,i])
    guess_amplitude = np.max(QDint[:,i])-np.min(QDint[:,i])
    guess_phase = (angle[int(np.where(QDint==QDint[:,i].max())[0])]-deg_0)*2*np.pi/180
    guess = [guess_amplitude, guess_phase, guess_min]
    data_first_guess = cos_sqr(np.linspace(0,np.pi*2,1000), *guess)

    '''
    Non-linear least square fit
    
    fit_prmt, pcov = curve_fit(cos_sqr, (angle-deg_0)*2*np.pi/180, QDint[:,i], p0=guess)    
    data_fit = cos_sqr(np.linspace(0,np.pi*2,1000), *fit_prmt)
    perr = np.sqrt(np.diag(pcov))
    '''
    
    
    '''
    Least square fit
    '''
    optimize_func = lambda x: x[0]*np.cos(x[1]-np.pi*(angle-deg_0)*2/180)**2 + x[2] - QDint[:,i]
    est_amplitude, est_phase, est_min = leastsq(optimize_func, guess)[0]
    

    data_fit = est_amplitude*np.cos(est_phase-np.linspace(0,np.pi*2,1000))**2 + est_min
    
    A = np.append(A, est_amplitude)
    phi = np.append(phi, est_phase)
    b = np.append(b,est_min)    
    
    
    ax[i].plot(np.linspace(0,360,1000), data_fit, color=acolor, linestyle='-', label='QD{}'.format(i)+' fit')
    ax[i].plot(np.linspace(0,360,1000), data_first_guess, color=acolor, linestyle='--', label='first guess')
    ax[i].set_xlim((angle.min()-deg_0)*2-1,(angle.max()-deg_0)*2+1)    
    ax[i].legend(fontsize='xx-small',frameon=None)
    fig.canvas.draw()

Imax = A+b
Imin = -(A-Imax)
Azimuth = phi*180/np.pi
delta = (Imax-Imin)/(Imax+Imin)
