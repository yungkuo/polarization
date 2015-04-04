# -*- coding: utf-8 -*-
"""
Created on Wed Apr 01 19:30:00 2015

@author: QCSE-adm
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from sympy.solvers import solve
from sympy import Symbol, sin, cos, asin
from sympy.mpmath import e



fig, ax = plt.subplots()
for i in range(8): 
    case = 2
    #deltatheta = 0.2
    NA = 1.4
    n1 = 1.5393
    n2 = 1.000293
    wavelength = 470# in unit of nm
    d = 1+i*100 # in unit of nm
    D = 0.1 #???
    
    theta1max = np.arcsin(NA/n1)
    theta1 = Symbol('theta1')    
    #theta1 = np.linspace(0,theta1max,1000)
    #def theta2(theta1,n1,n2):
    #    return np.arcsin(n1*np.sin(theta1)/n2)
    theta2 = asin(n1*sin(theta1)/n2)
    
    
    def delta(n1,d,theta1,wavelength):
        return 4*np.pi*n1*d*cos(theta1)/wavelength  
    def integrandA(theta1,D,nj,theta2,fs,fpb):
        return D**2*np.pi/4*nj*cos(theta2)/cos(theta1)*(np.abs(fs-cos(theta1)*fpb))**2*sin(theta1)
    def integrandB(theta1,D,nj,theta2,fpa):
        return D**2*np.pi*nj*cos(theta2)/cos(theta1)*(np.abs(fpa))**2*sin(theta1)**3
    def integrandC(theta1,D,nj,theta2,fpa,fs):
        return D**2*np.pi/2*nj*cos(theta2)/cos(theta1)*(np.abs(cos(theta1)*fpa+fs))**2*sin(theta1)
    def deltatheta(theta,A,B,C):
        return C*sin(theta)**2/((2*A-2*B+C)*sin(theta)**2+2*B)
    
    '''
    integrandA = lambda theta1: D**2*np.pi/4*nj*cos(theta2)/cos(theta1)*(np.abs(fs-cos(theta1)*fpb))**2*sin(theta1)
    integrandB = lambda theta1: D**2*np.pi*nj*cos(theta2)/cos(theta1)*(np.abs(fpa))**2*sin(theta1)**3
    integrandC = lambda theta1: D**2*np.pi/2*nj*cos(theta2)/cos(theta1)*(np.abs(cos(theta1)*fpa+fs))**2*sin(theta1)
    '''
    
    if case == 2:
        nj = n1
        rs12 = (n1*cos(theta1)-n2*cos(theta2))/(n1*cos(theta1)+n2*cos(theta2))
        rp12 = (n2*cos(theta1)-n1*cos(theta2))/(n1*cos(theta2)+n2*cos(theta1))
        fs = 1+rs12*e**(delta(n1,d,theta1,wavelength))
        fpa = 1+rp12*e**(delta(n1,d,theta1,wavelength))
        fpb = 1-rp12*e**(delta(n1,d,theta1,wavelength))
        '''
        A = np.nansum(integrandA(theta1,D,nj,theta2,fs,fpb)*(theta1max-0)/1000)
        B = np.nansum(integrandB(theta1,D,nj,theta2,fpa)*(theta1max-0)/1000)
        C = np.nansum(integrandC(theta1,D,nj,theta2,fpa,fs)*(theta1max-0)/1000)
        '''
       
        
        A = -1*np.diff(quad(integrandA, 0, theta1max, args=(D,nj,theta2,fs,fpb)))
        B = -1*np.diff(quad(integrandB, 0, theta1max, args=(D,nj,theta2,fpa)))
        C = -1*np.diff(quad(integrandC, 0, theta1max, args=(D,nj,theta2,fpa,fs)))
        #theta = Symbol('theta')
        #deltatheta = solve(C*sin(theta)**2/((2*A-2*B+C)*sin(theta)**2+2*B)-delta, theta)
        
        line = deltatheta(np.linspace(0,1.7,1000),A,B,C)    
        
        
        ax.plot(np.linspace(0,1.7*180/np.pi, 1000), line, label=i)
        ax.legend()
        y=[2.535455359520645269e-01,
            7.181671650422952191e-02,
            2.850634055605631989e-01,
            8.573893668366548704e-02,
            2.049022977713153282e-01,
            2.218994884859415706e-01,
            2.839161521363013363e-01,
            3.226611012534411627e-01,
            0.52]
        for i in range(len(y)):
            ax.axhline(y=y[i],xmin=0,xmax=1,c="y",linewidth=1,zorder=1, clip_on=False)
        
