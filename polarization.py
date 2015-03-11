# -*- coding: utf-8 -*-
"""
Created on Mon Mar 02 04:18:33 2015

@author: QCSE-adm
"""

import numpy as np
import matplotlib.pyplot as plt
import libtiff
from scipy.optimize import leastsq

filePath='E:/QCSE data/030715polarization/'
differential_image = 0 # 1 = yes, display differential image; else = no, display mean image

deg_0tif = libtiff.TiffFile(filePath+'semi0(167).tif')
deg_0 = np.array(deg_0tif.get_tiff_array()[:,:,:], dtype='d')
deg_15tif = libtiff.TiffFile(filePath+'semi15(182).tif')
deg_15 = np.array(deg_15tif.get_tiff_array()[:,:,:], dtype='d')
deg_30tif = libtiff.TiffFile(filePath+'semi30(197).tif')
deg_30 = np.array(deg_30tif.get_tiff_array()[:,:,:], dtype='d')
deg_45tif = libtiff.TiffFile(filePath+'semi45(212).tif')
deg_45 = np.array(deg_45tif.get_tiff_array()[:,:,:], dtype='d')
deg_60tif = libtiff.TiffFile(filePath+'semi60(227).tif')
deg_60 = np.array(deg_60tif.get_tiff_array()[:,:,:], dtype='d')
deg_75tif = libtiff.TiffFile(filePath+'semi75(242).tif')
deg_75 = np.array(deg_75tif.get_tiff_array()[:,:,:], dtype='d')
deg_90tif = libtiff.TiffFile(filePath+'semi90(257).tif')
deg_90 = np.array(deg_90tif.get_tiff_array()[:,:,:], dtype='d')
deg_153tif = libtiff.TiffFile(filePath+'semi(320).tif')
deg_153 = np.array(deg_153tif.get_tiff_array()[:,:,:], dtype='d')
deg_113tif = libtiff.TiffFile(filePath+'semi(280).tif')
deg_113 = np.array(deg_113tif.get_tiff_array()[:,:,:], dtype='d')

frame = len(deg_0[:,0,0])
row = len(deg_0[0,:,0])
col = len(deg_0[0,0,:])
scan = 5
if differential_image == 1:
    deg_0_df = np.zeros((row, col))
    for i in range(frame-1):       
        c=deg_0[i,:,:]
        d=deg_0[i+1,:,:]
        deg_0_df=deg_0_df+np.absolute(d-c) 
    deg_15_df = np.zeros((row, col))
    for i in range(frame-1):       
        c=deg_0[i,:,:]
        d=deg_0[i+1,:,:]
        deg_15_df=deg_15_df+np.absolute(d-c)
    deg_30_df = np.zeros((row, col))
    for i in range(frame-1):       
        c=deg_0[i,:,:]
        d=deg_0[i+1,:,:]
        deg_30_df=deg_30_df+np.absolute(d-c)
    deg_45_df = np.zeros((row, col))
    for i in range(frame-1):       
        c=deg_0[i,:,:]
        d=deg_0[i+1,:,:]
        deg_45_df=deg_45_df+np.absolute(d-c)
    deg_60_df = np.zeros((row, col))
    for i in range(frame-1):       
        c=deg_0[i,:,:]
        d=deg_0[i+1,:,:]
        deg_60_df=deg_60_df+np.absolute(d-c)
    deg_75_df = np.zeros((row, col))
    for i in range(frame-1):       
        c=deg_0[i,:,:]
        d=deg_0[i+1,:,:]
        deg_75_df=deg_75_df+np.absolute(d-c)
    deg_90_df = np.zeros((row, col))
    for i in range(frame-1):       
        c=deg_0[i,:,:]
        d=deg_0[i+1,:,:]
        deg_90_df=deg_90_df+np.absolute(d-c)    
        
    all_df = deg_0_df + deg_15_df + deg_30_df + deg_45_df + deg_60_df + deg_75_df + deg_90_df
    fig, ax = plt.subplots()
    ax.imshow(all_df)  

else: 
    deg_0_mean = np.mean(deg_0, axis=0)
    deg_15_mean = np.mean(deg_15, axis=0)
    deg_30_mean = np.mean(deg_30, axis=0)
    deg_45_mean = np.mean(deg_45, axis=0)
    deg_60_mean = np.mean(deg_60, axis=0)
    deg_75_mean = np.mean(deg_75, axis=0)
    deg_90_mean = np.mean(deg_90, axis=0)
    all_mean = deg_0_mean + deg_15_mean + deg_30_mean + deg_45_mean + deg_60_mean + deg_75_mean + deg_90_mean   
    fig, ax = plt.subplots()
    ax.imshow(all_mean)
    
pts = np.array(plt.ginput(n=0))
print pts
fig, ax = plt.subplots(len(pts))
for i in range(len(pts)):
    ax[i].imshow(all_mean[pts[i,1]-scan:pts[i,1]+scan, pts[i,0]-scan:pts[i,0]+scan])

def blinkon_mean(mov, pts, fig, ax):
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
        ax.plot(np.arange(0,frame,1), np.tile(threshold[i], frame), 'r-')
        fig.canvas.draw()
    
    QDintensity =[]
    for n in range(len(pts)):
        QD1 = np.mean([timetraces1[n,i] for i in range(frame) if timetraces1[n,i] > threshold[n]])
        ax.plot(np.arange(0,frame,1), np.tile(QD1, frame), 'y')
        fig.canvas.draw()
        QDintensity = np.append(QDintensity, QD1)
    
    return QDintensity

QDint_0 = blinkon_mean(deg_0, pts, fig, ax)
QDint_15 = blinkon_mean(deg_15, pts, fig, ax)
QDint_30 = blinkon_mean(deg_30, pts, fig, ax)
QDint_45 = blinkon_mean(deg_45, pts, fig, ax)
QDint_60 = blinkon_mean(deg_60, pts, fig, ax)
QDint_75 = blinkon_mean(deg_75, pts, fig, ax)
QDint_90 = blinkon_mean(deg_90, pts, fig, ax)
QDint_153 = blinkon_mean(deg_153, pts, fig, ax)
QDint_113 = blinkon_mean(deg_113, pts, fig, ax)
QDint = np.concatenate((QDint_0, QDint_15, QDint_30, QDint_45, QDint_60, QDint_75, QDint_90, QDint_113, QDint_153), axis=1)
QDint = np.reshape(QDint, [9, len(pts)])

def get_color():
    for item in ['r', 'g', 'b', 'c', 'm', 'y', 'k']:
        yield item
color = get_color()


fig, ax = plt.subplots()
angle = np.array([0,15,30,45,60,75,90,113,153])
arti_signal = np.reshape(np.tile(np.cos(np.pi*angle/180)**2, [len(pts)]),[9, len(pts)], order='F')
QDint = QDint + arti_signal*0
for i in range(len(pts)):
    acolor = next(color)    
    ax.scatter(angle, QDint[:,i], color=acolor, marker='o')
    
    guess_min = np.min(QDint[:,i])
    guess_amplitude = np.max(QDint[:,i])-np.min(QDint[:,i])
    guess_phase = 0
    
    data_first_guess = guess_amplitude*np.cos(np.linspace(0,np.pi,1000)+guess_phase)**2 + guess_min
    
    optimize_func = lambda x: x[0]*np.cos(np.pi*angle/180+x[1])**2 + x[2] - QDint[:,i]
    est_amplitude, est_phase, est_min = leastsq(optimize_func, [guess_amplitude, guess_phase, guess_min])[0]
    
    data_fit = est_amplitude*np.cos(np.linspace(0,np.pi,1000)+est_phase)**2 + est_min
    
    ax.plot(np.linspace(0,180,1000), data_fit, acolor+ '-', label='QD{}'.format(i)+' fit')
    #ax.plot(np.linspace(0,180,1000), data_first_guess, '--', label='first guess')
    ax.set_xlim(0,180)    
    ax.legend()
    fig.canvas.draw()