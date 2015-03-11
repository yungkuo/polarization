# -*- coding: utf-8 -*-
"""
Created on Sat Mar 07 05:48:21 2015

@author: QCSE-adm
"""

import numpy as np
import matplotlib.pyplot as plt
import libtiff

filePath='E:/QCSE data/030715polarization/'

bg = libtiff.TiffFile(filePath+'bg'+'.tif')
bg1 = bg.get_tiff_array()
bg2=np.array(bg1[:,:,:],dtype='d')

bg_no_monitor = libtiff.TiffFile(filePath+'bg no monitor'+'.tif')
bg_no_monitor1 = bg_no_monitor.get_tiff_array()
bg_no_monitor2=np.array(bg_no_monitor1[:,:,:],dtype='d')

bg_no_monitor_noLamp = libtiff.TiffFile(filePath+'bg no monitor no lamp1'+'.tif')
bg_no_monitor_noLamp1 = bg_no_monitor_noLamp.get_tiff_array()
bg_no_monitor_noLamp2=np.array(bg_no_monitor_noLamp1[:,:,:],dtype='d')

bg_no_monitor_noLamp_laserND = libtiff.TiffFile(filePath+'bg no monitor no lamp laserND'+'.tif')
bg_no_monitor_noLamp_laserND1 = bg_no_monitor_noLamp_laserND.get_tiff_array()
bg_no_monitor_noLamp_laserND2=np.array(bg_no_monitor_noLamp_laserND1[:,:,:],dtype='d')

bg_no_monitor_noLamp_noLaser = libtiff.TiffFile(filePath+'bg no monitor no lamp no laser2'+'.tif')
bg_no_monitor_noLamp_noLaser1 = bg_no_monitor_noLamp_noLaser.get_tiff_array()
bg_no_monitor_noLamp_noLaser2=np.array(bg_no_monitor_noLamp_noLaser1[:,:,:],dtype='d')

bg_mean = np.mean(np.mean(bg2, axis=1), axis=1)
bg_no_monitor_mean = np.mean(np.mean(bg_no_monitor2, axis=1), axis=1)
bg_no_monitor_noLamp_mean = np.mean(np.mean(bg_no_monitor_noLamp2, axis=1), axis=1)
bg_no_monitor_noLamp_laserND_mean = np.mean(np.mean(bg_no_monitor_noLamp_laserND2, axis=1), axis=1)
bg_no_monitor_noLamp_noLaser_mean = np.mean(np.mean(bg_no_monitor_noLamp_noLaser2, axis=1), axis=1)
bg_std = np.std(bg_mean,ddof=1,dtype='d')
bg_no_monitor_std = np.std(bg_no_monitor_mean,ddof=1,dtype='d')
bg_no_monitor_noLamp_std = np.std(bg_no_monitor_noLamp_mean,ddof=1,dtype='d')
bg_no_monitor_noLamp_laserND_std = np.std(bg_no_monitor_noLamp_laserND_mean,ddof=1,dtype='d')
bg_no_monitor_noLamp_noLaser_std = np.std(bg_no_monitor_noLamp_noLaser_mean,ddof=1,dtype='d')
print bg_std, bg_no_monitor_std, bg_no_monitor_noLamp_std, bg_no_monitor_noLamp_laserND_std, bg_no_monitor_noLamp_noLaser_std 

fig, ax = plt.subplots()
ax.plot(np.arange(300), bg_mean,'r', label='bg')
ax.plot(np.arange(300), bg_no_monitor_mean,'b', label=' bg_no_monitor')
ax.plot(np.arange(300), bg_no_monitor_noLamp_mean,'k', label='bg_no_monitor_noLamp')
ax.plot(np.arange(300), bg_no_monitor_noLamp_laserND_mean,'c', label='bg_no_monitor_noLamp_laserND')
ax.plot(np.arange(300), bg_no_monitor_noLamp_noLaser_mean,'g', label='bg_no_monitor_noLamp_noLaser')
ax.plot(np.arange(300), bg_std+bg_mean,'r')
ax.plot(np.arange(300), -bg_std+bg_mean,'r')
ax.plot(np.arange(300), bg_no_monitor_std+bg_no_monitor_mean,'b')
ax.plot(np.arange(300), -bg_no_monitor_std+bg_no_monitor_mean,'b')
ax.plot(np.arange(300), bg_no_monitor_noLamp_std+bg_no_monitor_noLamp_mean,'k')
ax.plot(np.arange(300), -bg_no_monitor_noLamp_std+bg_no_monitor_noLamp_mean,'k')
ax.plot(np.arange(300), bg_no_monitor_noLamp_laserND_std+bg_no_monitor_noLamp_laserND_mean,'c')
ax.plot(np.arange(300), -bg_no_monitor_noLamp_laserND_std+bg_no_monitor_noLamp_laserND_mean,'c')
ax.plot(np.arange(300), bg_no_monitor_noLamp_noLaser_std+bg_no_monitor_noLamp_noLaser_mean,'g')
ax.plot(np.arange(300), -bg_no_monitor_noLamp_noLaser_std+bg_no_monitor_noLamp_noLaser_mean,'g')


handles, labels = ax.get_legend_handles_labels()    
ax.legend(handles, loc=2, borderaxespad=0, fontsize=10)