#!/usr/bin/python
from scipy.integrate import odeint
import matplotlib.pyplot as plt # for plotting          
import numpy as np

class Particle(object):

    """Class that describes particle"""
    m = 1.0

    def __init__(self, x0=1.0, y0 =0.0, z0= 0.00, u0=0.0, v0 = 0.0, w0 = 0.0,  tf = 10.0, dt = 0.01):
        # print("particle init'd")
        self.x = np.array([x0,y0,z0])
        self.v = np.array([u0,v0,w0])
        self.t = 0.0
        self.tf = tf
        self.dt = dt
        npoints = int(tf/dt) # always starting at t = 0.0
        self.npoints = npoints
        self.tarray = np.linspace(0.0, tf,npoints, endpoint = True) # include final timepoint
        self.xv0 = np.ravel(np.array([self.x, self.v])) # NumPy array with initial position and velocity

    def reinitialize(self):
        self.npoints = int(self.tf/self.dt)
        self.x = self.xv0[0:3]
        self.v = self.xv0[3:]
        self.t = 0

    
    def F(self, x, v, t):
        return np.array([0.0, 0.0, 0.0])

    def Euler_step(self): # increment position as before
        a = self.F(self.x, self.v, self.t) / self.m
        self.x += self.v * self.dt
        self.v += a * self.dt
        self.t += self.dt
    
    def RK4_step(self):
        a1 = self.F(self.x, self.v, self.t) / self.m
        
        k1 = np.array([self.v, a1])*self.dt

        a2 = self.F(self.x+k1[0]/2, self.v+k1[1]/2, self.t+self.dt/2) / self.m
        k2 = np.array([self.v+k1[1]/2 ,a2])*self.dt
        
        a3 = self.F(self.x+k2[0]/2, self.v+k2[1]/2, self.t+self.dt/2) / self.m
        k3 = np.array([self.v+k2[1]/2, a3])*self.dt
        
        a4 = self.F(self.x+k3[0], self.v+k3[1], self.t+self.dt) / self.m
        k4 = np.array([self.v+k3[1], a4])*self.dt

        self.x += (k1[0]+ k4[0])/6 + (k2[0] + k3[0])/3
        self.v += (k1[1]+ k4[1])/6 + (k2[1] + k3[1])/3
        
        self.t += self.dt

    def Euler_trajectory(self):  # calculate trajectory as before
        # will reinitialize euler trajectory everytime this method is called
        x_euler = np.zeros([self.npoints, 3])
        v_euler = np.zeros([self.npoints, 3])

        for ii in range(self.npoints):
            x_euler[ii] = self.x
            v_euler[ii] = self.v
            self.Euler_step()
        
        self.x_euler = x_euler
        self.v_euler = v_euler
    
    def RK4_trajectory(self):  # calculate trajectory as before
        # need to reinitialize if you want to call this method and others
        x_RK4 = np.zeros([self.npoints, 3])
        v_RK4 = np.zeros([self.npoints, 3])
        
        for ii in range(self.npoints):
            x_RK4[ii] = self.x
            v_RK4[ii] = self.v
            self.RK4_step()

        self.x_RK4 = x_RK4
        self.v_RK4 = v_RK4

    def scipy_trajectory(self):
        """calculate trajectory using SciPy ode integrator"""
        self.xv = odeint(self.derivative, self.xv0, self.tarray)

    def derivative(self, xv, t):
        """right hand side of the differential equation"""
        x = np.array([xv[0], xv[1], xv[2]])
        v = np.array([xv[3], xv[4], xv[5]])
        a = self.F(x, v, t) / self.m
        return np.ravel(np.array([v, a]))

    def results(self):
        print('\n\t Position and Velocity at Final Time:')
        print('\t Euler:')
        print('\t t = {0:0.2f} | r = [{1:0.2f}, {2:0.2f}, {3:0.2f}] | v = [{4:0.2f}, {5:0.2f}, {6:0.2f}]'.format(self.t, *self.x , *self.v))
        
        if hasattr(self, 'xv'):
            print('\t SciPy ODE Integrator:')
            print('\t t = {0:0.2f} r = [{1:0.2f}, {2:0.2f}, {3:0.2f}] v = [{4:0.2f}, {5:0.2f}, {6:0.2f}]'.format(self.tarray[-1], *self.xv[-1,0:3] , *self.xv[-1,3:]))

    def plot3D(self, t, r, v):
        fig = plt.figure(figsize = [8.5,3])
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        ax1.plot(t,r[:,0],'k')
        ax2.plot(t,r[:,1],'k')
        ax3.plot(t,r[:,2],'k')
    
        ax1.set_xlabel("t")
        ax2.set_xlabel("t")
        ax3.set_xlabel("t")
        
        ax1.set_ylabel("x")
        ax2.set_ylabel("y")
        ax3.set_ylabel("z")
        fig.subplots_adjust(left = 0.1, right = 0.96, bottom = 0.16, wspace = 0.4)

        fig = plt.figure(figsize = [7.5,3])
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        ax1.plot(r[:,0],r[:,1],'k')
        ax2.plot(r[:,0],r[:,2],'k')
        ax3.plot(r[:,1],r[:,2],'k')
    
        ax1.set_xlabel("x")
        ax2.set_xlabel("x")
        ax3.set_xlabel("y")
        
        ax1.set_ylabel("y")
        ax2.set_ylabel("z")
        ax3.set_ylabel("z")

        fig.subplots_adjust(left = 0.1, right = 0.96, bottom = 0.16, wspace = 0.4)

    def plot(self):
        if hasattr(self,'xv'):
            self.plot3D(self.tarray, self.xv[:, 0:3],self.xv[:,3:])
        if hasattr(self,'x_euler'):
            self.plot3D(self.tarray, self.x_euler,self.v_euler)
        if hasattr(self,'x_RK4'):
            self.plot3D(self.tarray, self.x_RK4, self.v_RK4)
            
class Projectile(Particle):

    """Subclass of Particle Class that describes a falling particle"""

    def __init__(self, m = 1.0, Cd = 0.5, x0 = 0.0, y0 = 0.0, z0 = 1.0 , u0 = 0.0, v0 = 0.0, w0 = 0.0, tf = 10.0,  dt = 0.001):
        # print("projectile init'd")
        self.m = m
        self.Cd = Cd


        super().__init__(x0,y0,z0,u0,v0,w0,tf,dt)   # call initialization method of the super (parent) class
   

          
    def F(self, x, v, t):
        g = 9.8
        # set sign of drag always opposite to velocity
        # and take care of division by zero, could have also just used np.sign(v)
        # but this way demonstrates 'list comprehension'
        # this is a faster way to construct a list than an explicit for loop


        v_hat = np.array([np.abs(vi)/vi if vi else 0 for vi in v])
        mod_v = np.sqrt(np.sum(v**2))
         
         
        Drag = -self.Cd*v_hat*mod_v*v
        G = np.array([0,0,-self.m*g])

        return G+Drag

    # overload method to prevent negative z (earths surface)
    def scipy_trajectory(self):
        Particle.scipy_trajectory(self)
        
        # set z = 0 as the earth's surface
        self.xv[np.nonzero(self.xv[:,2]<0),2] = 0.0


class Rotating_Projectile(Particle):

    """Subclass of Particle Class that describes a falling rotating particle"""

    def __init__(self, m = 1.0, r = 1, x0 = 0.0, y0 = 0.0, z0 = 1.0 , u0 = 0.0, v0 = 0.0, w0 = 0.0, i0 = 0, j0 = 0, k0 = 0, tf = 10.0,  dt = 0.001):
        # print("projectile init'd")
        self.m = m
        self.omega = np.array([i0,j0,k0])
        self.r = r
        self.A = np.pi*r**2   # cross-secitonal area
        self.D = 1

        super().__init__(x0,y0,z0,u0,v0,w0,tf,dt)   # call initialization method of the super (parent) class
   

    def density(self,x):
        T0 = 300 
        rho0 = 1.225
        a = 6.5e-3
        alpha = 2.5

        if x[-1] < 2.5e4:
            rho = rho0*(1-a*x[-1]/T0)**alpha
        else :
            rho = rho0*(1-a*2.5e4/T0)**alpha
        return 1 


    def Cd(self,mod_v):
        vc, a, b, c = 10, 0.25, 0.25, 0.16

        chi = (mod_v - vc)/4.
        
        if mod_v < 1e-6 :
            return 0
    
        if  chi <= 0:
            factor = np.exp(-chi**2)
        elif chi > 0:
            factor = np.exp(-chi**2/4)

        Drag_Coeff = a + b/(1+np.exp(chi)) - c*factor

        return Drag_Coeff
    
    def omega(self,t):
        tau = 1
        return self.omega0*np.exp(-t/tau)

    def lift(self,x, v,t):
        
        mod_omega = np.sqrt(np.sum(self.omega**2))
        mod_v = np.sqrt(np.sum(v**2))
        
        # cut-off at small omega to avoid division by zero 
        if mod_omega < 1e-6:
            Fl = 0
        else :

            Cl = 0.5*(self.r*mod_omega/mod_v)**(0.4)
            Fl = 0.5*self.A*self.r*np.cross(self.omega,v)

        return Fl

      
    def F(self, x, v, t):
        g = 9.8
        # set sign of drag always opposite to velocity
        # and take care of division by zero, could have also just used np.sign(v)
        # but this way demonstrates 'list comprehension'
        # this is a faster way to construct a list than an explicit for loop
        # v_hat = np.array([np.abs(vi)/vi if vi else 0 for vi in v])
        mod_v = np.sqrt(np.sum(v**2))
        G = np.array([0,0,-self.m*g])
        
        
        if mod_v < 1e-6:
            # no reason to calculate drag or lift for very small velocities.
            return G
 
        Drag = -self.D*self.Cd(mod_v)*self.density(x)*mod_v*v
        Lift = self.FL(x,v,t)

        return G+Drag+Lift

    # overload method to prevent negative z (earths surface)
    def scipy_trajectory(self):
        Particle.scipy_trajectory(self)
        
        # set z = 0 as the earth's surface
        self.xv[np.nonzero(self.xv[:,2]<0),2] = 0.0



class ChargedParticle(Particle):

    """Subclass of Particle Class that describes particle in E and B fields"""
    
    E = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, 1.0])

    def __init__(self, m = 0.1057, q = 1.0, x0 = 0.0, y0 = 0.0, z0 = 1.0 , v0 = 0.0, u0 = 0.0, w0 = 0.0,tf = 10.0,  dt = 0.1):
        
        self.m = m  
        self.q = q
        super().__init__(x0,y0,z0,v0,u0,w0,tf,dt)       


    def F(self, x, v, t):
        return self.q * (self.E + np.cross(v, self.B))


class RelativisticChargedParticle(Particle):

    """Subclass of Particle Class that describes a relativistic particle in E and B fields"""
    
    # Using GeV and natural units: hbar = c = 1
    #  1/eV = 1.97e-7 m
    #  1/GeV = 0.6582 micro-seconds
    #  1/eV**2 = 1.444 mT 
    
    TeslatoeV2 = 1.444027e-3
    MetertoeV = 1.97e-2

    E = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, 0.0])

    def __init__(self, m = 0.1057, q = 1.0, x0 = 0.0, y0 = 0.0, z0 = 0.0 , u0 = 0.0, v0 = 0.0, w0 = 0.0,tf = 1.0,  dt = 0.0001):
        self.q = q #charge is 1 in natural units
        self.m = m  #rest mass in units of GeV
        
        super().__init__(x0,y0,z0,u0,v0,w0,tf,dt)       

    def gamma(self):
        # units where c = 1

        modv2 = np.sum(self.v**2) 
        
        return 1/np.sqrt(1-modv2)

    def F(self, x, v, t):

        r = np.sqrt(x[0]**2 +x[1]**2)
                
        if r > 4.25*self.MetertoeV and r < 11*self.MetertoeV :

            self.B = np.array([0.0, 4, 0.0])*self.TeslatoeV2#*1e9#*1e9

            FB = self.q * (self.E + np.cross(v, self.B))
           
            # relativistic correction

            F = 1./self.gamma()*(FB-np.dot(v,FB)*v)
        else :
            F =np.array([0.0, 0.0, 0.0]) 
          
        return F


    def plot(self):
        
        fig = plt.figure(figsize = [8.5,3])
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        ax1.plot(self.tarray,self.xv[:,0]/self.MetertoeV)
        ax2.plot(self.tarray,self.xv[:,1]/self.MetertoeV)
        ax3.plot(self.tarray,self.xv[:,2]/self.MetertoeV)


        ax1.fill_between(np.linspace(0,self.tf), 11,4.25, alpha = 0.5, color = 'grey')
        ax1.hlines([4.25, 11],0,self.tf)
        
        ax1.set_xlabel("t")
        ax2.set_xlabel("t")
        ax3.set_xlabel("t")
        
        ax1.set_ylabel("x (m)")
        ax2.set_ylabel("y (m)")
        ax3.set_ylabel("z (m)")

        fig.subplots_adjust(left = 0.1, right = 0.96, bottom = 0.16, wspace = 0.4)

        fig = plt.figure(figsize = [4,4])

        ax1 = fig.add_subplot(111)

        r = np.sqrt(np.sum(self.xv[:,0:2]**2, axis = 1))/self.MetertoeV

        ax1.plot(self.xv[:,2]/self.MetertoeV, r, lw = 2)
        ax1.plot(np.zeros(100), np.linspace(0,12,100), ls = '--', color = 'grey')
        ax1.fill_between(np.linspace(-2.5,2.5), 11,4.25, alpha = 0.5, color = 'grey')
        ax1.hlines([4.25, 11],-2.5,2.5)
        

        limits = self.xv[-1,2]/self.MetertoeV
        ax1.set_xlabel("z (m)")
        ax1.set_ylabel("r (m)")
        ax1.set_xlim([-limits-1e-6, limits+1e-6])
        ax1.set_ylim([0, 12])

        plt.show()





