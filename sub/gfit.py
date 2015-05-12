# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:31:58 2015

@author: yung
"""

import numpy as np
import matplotlib.pyplot as plt
import lmfit
import libtiff
from mpl_toolkits.axes_grid1 import make_axes_locatable

def Gaussian(x, a, x0, sigma, b):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))+b

def find_absigma(mov, scan, pts):
    a = []
    b = []
    sigma = []
    for i in range(len(pts)):
        box = np.mean(mov[:, pts[i,1]-scan:pts[i,1]+scan, pts[i,0]-scan:pts[i,0]+scan], axis=0)
        xmean = np.mean(box, axis=0)
        ymean = np.mean(box, axis=1)
        x = np.arange(0,scan*2,1)
        
        fig, ax = plt.subplots(2,2, sharex=True, figsize=(8,10))
        fig.tight_layout(pad=3, w_pad=10, h_pad=3)
        ax[0,0] = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        ax[1,1] = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
        ax[1,0] = plt.subplot2grid((3, 3), (2, 0), colspan=2)
        im = ax[0,0].imshow(box, interpolation = 'none', aspect='auto')
        divider = make_axes_locatable(ax[0,0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        gmod = lmfit.Model(Gaussian)
        params = gmod.make_params()
        for j in enumerate([xmean,ymean]):
            params['a'].set(value = j[1].max()-j[1].min(), min=0)
            params['x0'].set(value = np.mean(x[np.where(j[1]==j[1].max())]))
            params['sigma'].set(value = len(j[1])/4, max=len(j[1]))
            params['b'].set(value = j[1].min(), min=0)
            #print params
            result = gmod.fit(j[1], x=x, **params)
            aj = result.best_values['a']
            bj = result.best_values['b']
            sigmaj = result.best_values['sigma']     
            
            if j[0] == 0:        
                result.plot_fit(ax=ax[1,j[0]], numpoints=100)
                ax[1,j[0]].set_xlabel('Pixels')
                ax[1,j[0]].set_ylabel('Intensity')
                ax[1,0].legend(bbox_to_anchor=(1.5, 0.5), ncol=1, frameon=False, fontsize = 'xx-small')
            
            else:
                ax[1,j[0]].plot(j[1], x, 'ro')
                ax[1,j[0]].plot(Gaussian(np.linspace(0,scan*2,100), aj, result.best_values['x0'], sigmaj, bj), np.linspace(0,scan*2,100), 'g-')
                ax[1,j[0]].plot(Gaussian(np.linspace(0,scan*2,100), j[1].max()-j[1].min(), np.mean(x[np.where(j[1]==j[1].max())]), len(j[1])/4, j[1].min()), np.linspace(0,scan*2,100), 'b--')            
                ax[1,j[0]].set_xlim([j[1].min(), j[1].max()])
                ax[1,j[0]].set_ylim([x.max(), x.min()])
                #ax[1,j[0]].tick_params(axis='x', labelsize=10)
                for tick in ax[1,j[0]].get_xticklabels():
                    tick.set_rotation(50)            
                ax[1,j[0]].set_xlabel('Intensity')
                ax[1,j[0]].set_ylabel('Pixels')
           
            a = np.append(a, aj)
            b = np.append(b, bj)
            sigma = np.append(sigma, sigmaj)            
            #print [scan,a,b,sigma]
    a = np.reshape(a, [len(pts),2])
    b = np.reshape(b, [len(pts),2])
    sigma = np.reshape(sigma, [len(pts),2])
    return a, b, sigma
    
    

f = '/Users/yung/run4/167.tif'
mov = libtiff.TiffFile(f)
mov = np.array(mov.get_tiff_array()[:,:,:], dtype='d')
pts = np.array([[ 246.55590062,  123.31987578],[ 260.86273292,  377.73602484],
                [ 195.87267081, 114.39440994], [ 206.80434783,  366.79813665],
                [ 228.68012422,  193.89130435],[ 241.58695652,  452.25776398]])
a, b, sigma = find_absigma(mov, 5, pts)
print a
print b
print sigma