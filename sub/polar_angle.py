# -*- coding: utf-8 -*-
"""
Created on Wed Apr 01 19:30:00 2015

@author: QCSE-adm
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from sympy.solvers import solve
from sympy import Symbol
from sympy import sin


fig, ax = plt.subplots()
for i in range(10): 
    delta = 0.2
    NA = 0.7+i*0.1
    n1 = 1.5
    n2 = 1.5
    wavelength = 470 # in unit of nm
    d = 20 # in unit of nm
    D = 1
    theta1 = np.arcsin(NA/n1)
    theta2 = np.arcsin(n1*np.sin(theta1)/n2)
    delta = 4*np.pi*n1*d*np.cos(theta1)/wavelength
    case = 2
    
    def integrandA(theta,D,n1,nj,theta2,fs,fpb):
        return D**2*np.pi/4*nj*np.cos(theta2)/np.cos(theta)*(np.abs(fs-np.cos(theta)*fpb))**2*np.sin(theta)
    def integrandB(theta,D,nj,theta2,fpa):
        return D**2*np.pi*nj*np.cos(theta2)/np.cos(theta)*(np.abs(fpa))**2*np.sin(theta)**3
    def integrandC(theta,D,nj,theta2,fpa,fs):
        return D**2*np.pi/2*nj*np.cos(theta2)/np.cos(theta)*(np.abs(np.cos(theta)*fpa+fs))**2*np.sin(theta)
    def deltatheta(theta,A,B,C):
        return C*np.sin(theta)**2/((2*A-2*B+C)*np.sin(theta)**2+2*B)
    
    if case == 2:
        nj = n2
        rs12 = (n1*np.cos(theta1)-n2*np.cos(theta2))/(n1*np.cos(theta1)+n2*np.cos(theta2))
        rp12 = (n2*np.cos(theta1)-n1*np.cos(theta2))/(n1*np.cos(theta2)+n2*np.cos(theta1))
        fs = 1+rs12*np.exp(delta*1J)
        fpa = 1+rp12*np.exp(delta*1J)
        fpb = 1-rp12*np.exp(delta*1J)
        A = -1*np.diff(quad(integrandA, 0, theta1, args=(D,n1,nj,theta2,fs,fpb)))
        B = -1*np.diff(quad(integrandB, 0, theta1, args=(D,nj,theta2,fpa)))
        C = -1*np.diff(quad(integrandC, 0, theta1, args=(D,nj,theta2,fpa,fs)))
        theta = Symbol('theta')
        #deltatheta = solve(C*sin(theta)**2/((2*A-2*B+C)*sin(theta)**2+2*B)-delta, theta)
        line = deltatheta(np.linspace(0,1.7,1000),A,B,C)    
        
        
        ax.plot(np.linspace(0,1.7, 1000), line, label=i)
        ax.legend()
       

