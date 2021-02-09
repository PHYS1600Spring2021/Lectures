#!/usr/bin/python
from scipy.integrate import odeint
import matplotlib.pyplot as plt # for plotting          
import numpy as np
from copy import copy

class Particle (object):

    """Class that describes particle"""
    m = 1.0

    def __init__(self, x0=1.0, v0=0.0,  tf = 10.0, dt = 0.001):
        self.x = x0
        self.v = v0
        self.x0 = x0
        self.v0 = v0
        self.t = 0.0
        self.tf = tf
        self.dt = dt

        self.tlabel = 'time (s)'
        self.xlabel = 'x (m)'
        self.vlabel = 'v (m/s)'

        npoints = int(tf/dt) # always starting at t = 0.0
        self.npoints = npoints
        self.tarray = np.linspace(0.0, tf,npoints, endpoint = True) # include final timepoint
        self.xv0 = np.array([self.x, self.v]) # NumPy array with initial position and velocity

    def F(self, x, v, t):
        # The force on a free particle is 0
        return array([0.0])

    def Euler_step(self): 
        """
        Take a single time step using Euler method
        """
        
        a = self.F(self.x, self.v, self.t) / self.m
        self.x += self.v * self.dt
        self.v += a * self.dt
        self.t += self.dt


    def Verlet_step(self):
         
        a = self.F(self.x, self.v, self.t)/self.m
        self.x += self.v * self.dt + 0.5 * self.dt**2 * a
        
        a2 = self.F(self.x, self.v, self.t)/self.m    
        self.v += 0.5 * self.dt * (a + a2)

        self.t +=self.dt


    def Euler_trajectory(self):
        """
        Loop over all time steps to construct a trajectory with Euler method
        Will reinitialize euler trajectory everytime this method is called
        """
        
        x_euler = []
        v_euler = []
        self.x = self.x0
        self.v = self.v0
        self.t = 0.0

        
        for ii in range(self.npoints):
            v_euler.append(self.v)
            x_euler.append(self.x)
            self.Euler_step()
        
        self.x_euler = np.array(x_euler)
        self.v_euler = np.array(v_euler)

    def verlet_trajectory(self):
        """
        Loop over all time steps to construct a trajectory with verlet method
        Will reinitialize verlet trajectory everytime this method is called
        """
        
        x_verlet = []
        v_verlet = []
        self.x = self.x0
        self.v = self.v0
        self.t = 0.0
        
        for ii in range(self.npoints):
            v_verlet.append(self.v)
            x_verlet.append(self.x)
            self.Verlet_step()
        
        self.x_verlet = np.array(x_verlet)
        self.v_verlet = np.array(v_verlet)


    def scipy_trajectory(self):
        """calculate trajectory using SciPy ode integrator"""
        self.xv = odeint(self.derivative, self.xv0, self.tarray)

    def derivative(self, xv, t):
        """right hand side of the differential equation"""
        x =xv[0]
        v =xv[1]
        a = self.F(x, v, t) / self.m
        return np.ravel(np.array([v, a]))

    def results(self):
        """
        Print out results in a nice format
        """

        print('Initial: t = 0.0, x = {}, v = {}, dt = {}'.format(self.x0 , self.v0, self.dt))
        print('Current: t = {}, x = {}, v = {}'.format(self.t, self.x , self.v))

        if hasattr(self, 'xv'):
            print('SciPy ODE Integrator:')
            print('t = {} x = {} v = {}'.format(self.tarray[-1], self.xv[-1, 0], self.xv[-1,1]))

    def plot(self, pt = 'trajectory', axs = None):
        """
        Make nice plots of our results
        """
        if not axs:
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
        
        
        if hasattr(self,'xv'):

            if pt == 'trajectory':
                if not axs:
                    ax1.plot(self.tarray, self.xv[:, 0], "k", label = 'odeint')
                else:
                    axs.plot(self.tarray, self.xv[:, 0], "k", label = 'odeint')

            if pt == 'phase':
                if not axs:
                    ax1.plot(self.xv[:, 0], self.xv[:, 1], "k",'.', label = 'odeint')
                else:
                    axs.plot(self.xv[:, 0], self.xv[:, 1], "k",'.', label = 'odeint')

        if hasattr(self,'x_euler'):

            if pt == 'trajectory':
                if not axs:
                    ax1.plot(self.tarray, self.x_euler, "b", label = 'euler')
                else:
                    axs.plot(self.tarray, self.x_euler, "b", label = 'euler')

            if pt == 'phase':
                if not axs:
                    ax1.plot(self.x_euler, self.v_euler, "b",'.', label = 'euler')
                else:
                    axs.plot(self.x_euler, self.v_euler, "b",'.', label = 'euler')
        
        if hasattr(self,'x_verlet'):

            if pt == 'trajectory':
                if not axs:
                    ax1.plot(self.tarray, self.x_verlet, "r", label = 'verlet')
                else:
                    axs.plot(self.tarray, self.x_verlet, "r", label = 'verlet')

            if pt == 'phase':
                if not axs:
                    ax1.plot(self.x_verlet, self.v_verlet, "r",'.', label = 'verlet')
                else:
                    axs.plot(self.x_verlet, self.v_verlet, "r",'.', label = 'verlet')
        if not axs:
            if pt == 'trajectory':
                ax1.set_xlabel(self.tlabel)
                ax1.set_ylabel(self.xlabel)
        
            if pt == 'phase':
                ax1.set_xlabel(self.xlabel)
                ax1.set_ylabel(self.vlabel)

            ax1.legend()

class Pendulum(Particle):

    """Subclass of Particle Class that describes a pendulum in a harmonic potential"""
    def __init__(self, l = 9.8, m = 1.0, x0 = 0.0 ,v0 = 0.0, tf = 50.0, dt = 0.001):
       
        super().__init__(x0,v0,tf,dt) 
        # for pendulum x = theta [-pi, pi]
        g = 9.8
        omega0 = np.sqrt(g/l)
        
        self.l = l # length
        self.m = m # mass
        self.omega0 = omega0 # natural frequency

        self.tlabel = 'time ($1/\omega_0$)'
        self.xlabel = '$\\theta$ (radians)'
        self.vlabel = '$\omega$ (radians/s)'

    def Euler_step(self): 
        # overload method to wrap x between [-pi,pi]
        
        Particle.Euler_step(self)
        if self.x > np.pi:
            self.x -= 2*np.pi
        elif self.x < -np.pi:
            self.x += 2*np.pi
 
    def verlet_step(self):
        # overload method to wrap x between [-pi,pi]
        
        Particle.verlet_step(self)
        if self.x > np.pi:
            self.x -= 2*np.pi
        elif self.x < -np.pi:
            self.x += 2*np.pi
 

    # overload method to wrap x between [-pi,pi]
    def scipy_trajectory(self):
        Particle.scipy_trajectory(self)
        
        x = self.xv[:,0]
        x_new = np.zeros(np.shape(x))
        x_new[0] = x[0]

        # find change in x between each point
        dx = np.diff(x)
        nx = np.shape(x)[0]
        
        for ii in range(1,nx):
            # reconstruct x array, checking for out of range values
            x_new[ii] = x_new[ii-1]+dx[ii-1]
            if x_new[ii] > np.pi:
                x_new[ii] -= 2*np.pi
            
            elif x_new[ii] < -np.pi:
                x_new[ii] += 2*np.pi
        
        self.xv_unwrap = copy(self.xv)
        self.xv[:,0] = x_new
    

    def F(self, x, v, t):
        g = 9.8 
        F = - g/self.l*np.sin(x)
        
        return F

            
