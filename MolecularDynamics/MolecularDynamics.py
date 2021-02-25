#!/Users/name/anaconda/bin/python3
import numpy as np 
# import Potentials
import matplotlib.pyplot as plt
import matplotlib.animation as am
from matplotlib import colors, cm
from numba import jit

# METHODS FOR FORCE CALCULATION, REMOVED FROM MAIN CLASS TO SPEED UP WITH JIT
@jit(nopython=True)
def jit_lennardJonesForce(N, L, x, y):

    f = np.zeros((2*N), np.double) 
    virial = 0
    tiny = 1.0e-40
    halfL = L/2
    
    cutoff = L/3

    for ii  in range(N):
        for jj in range(ii+1, N):
            dx = x[ii] - x[jj]
            if( dx > halfL): dx = dx - L
            if( dx < -halfL): dx = dx + L

            dy = y[ii] - y[jj]
            if( dy > halfL): dy = dy - L
            if( dy < -halfL): dy = dy + L

            if dx**2+dy**2 > cutoff:
                continue
            
            r2inv = 1.0/(dx*dx + dy*dy + tiny)
            c = 48.0 * r2inv**7 - 24.0 * r2inv**4
            fx = dx * c
            fy = dy * c
            
            f[2*ii] += fx;
            f[2*ii+1] += fy;
            f[2*jj] -= fx; # Newton's 3rd law
            f[2*jj+1] -= fy;
            
            virial += fx*dx + fy*dy; # for virial accumulator (calculation of pressure)
                    
    return f, virial


@jit(nopython = True)
def jit_lennardJonesPotentialEnergy(N, L, x, y):        

    tiny = 1.0e-40
    halfL = L/2
    U = 0.0
    cutoff = L/3
    # r_pair = np.zeros((2*N), np.double)
    
    for ii in range(N):  # The O(N**2) calculation that slows everything down
        for jj in range(ii+1,N): 

            dx = x[ii] - x[jj]
            if( dx > halfL): dx = dx - L
            if( dx < -halfL): dx = dx + L

            dy = y[ii] - y[jj]
            if( dy > halfL): dy = dy - L
            if( dy < -halfL): dy = dy + L

            if dx*dx+dy*dy > cutoff:
                continue
            
            r2inv = 1.0/(dx*dx + dy*dy + tiny)
            dU = r2inv**6 - r2inv**3
            U+=dU

    return 2.0 * U


@jit(nopython = True)
def jit_PowerLawForce(N, L, x, y): 

    f = np.zeros((2*N), np.double) 
    virial = 0
    tiny = 1.0e-40
    halfL = L/2

    ii = 0
    while (ii <  N):

        jj = ii+1
        while (jj < N):
            dx = x[ii] - x[jj]
            if( dx > halfL): dx = dx - L
            if( dx < -halfL): dx = dx + L

            dy = y[ii] - y[jj]
            if( dy > halfL): dy = dy - L
            if( dy < -halfL): dy = dy + L

            r2inv = 1.0 / (dx*dx + dy*dy + tiny)
            r6inv = r2inv*r2inv*r2inv
            fx = dx * r6inv
            fy = dy * r6inv;

            f[2*ii] += fx;
            f[2*ii+1] += fy;
            f[2*jj] -= fx; # Newton's 3rd law
            f[2*jj+1] -= fy;
                            
            virial += fx*dx + fy*dy; # for virial accumulator (calculation of pressure)
            jj+=1
        ii+=1
            
    return f, virial 


class MolecularDynamics:
    """Class that describes the molecular dynamics of a gas of atoms in units where m = epsilon = sigma = kb = 1"""
    
    # when to take measurements (every 100 timesteps)
    sampleInterval = 100
    
    def __init__(self, N=4, L=10.0, initialTemperature=0.0, dt = 0.001, forceType = "lennardJones"):

        # numpy.random.seed(219) # random number generator used for initial velocities (and sometimes positions) 
        
        self.N = N  # number of particles 
        self.L = L      # length of square side 
        self.initialTemperature = initialTemperature
        
        self.t = 0.0 # initial time
        self.dt = dt
        self.tArray = np.array([self.t]) # array of time steps that is added to during integration
        self.steps = 0
        
        self.EnergyArray = np.array([]) # list of energy, sampled every sampleInterval time steps
        self.sampleTimeArray = np.array([])
        
        # accumulate statistics during time evolution
        self.temperatureArray = np.array([self.initialTemperature])
        self.temperatureAccumulator = 0.0
        self.squareTemperatureAccumulator = 0.0
        self.virialAccumulator = 0.0

        self.x = np.zeros(2*N) # NumPy array of N (x, y) positions
        self.v = np.zeros(2*N) # array of N (vx, vy) velocities

        self.xArray = np.array([]) # particle positions that is added to during integration
        self.vArray = np.array([]) # particle velocities
        
        self.forceType = forceType


    def minimumImage(self, x): # minimum image approximation (Gould Listing 8.2)

        L = self.L
        halfL = 0.5 * L
        
        return (x + halfL) % L - halfL


    def force(self): 

        if (self.forceType == "lennardJones"):
            f, virial = self.lennardJonesForce()
        
        if (self.forceType == "lennardJones_jit"):
            f, virial = self.lennardJonesForce_jit()
                
        if (self.forceType == "powerLaw"):
            f, virial = self.powerLawForce()
        
        if (self.forceType == "powerLaw_jit"):
            f, virial = self.powerLawForce_jit()

        self.virialAccumulator += virial
        
        return f
            
    
    def lennardJonesForce(self): # Gould Eq. 8.3 (NumPy vector form which is faster)

        N = self.N
        virial = 0.0
        tiny = 1.0e-40 # prevents division by zero in calculation of self-force
        L = self.L
        halfL = 0.5 * L
        
        x = self.x[np.arange(0, 2*N, 2)]
        y = self.x[np.arange(1, 2*N, 2)]   
        f = np.zeros(2*N)
        
        minimumImage = self.minimumImage
        
        for i in range(N):  # The O(N**2) calculation that slows everything down
    
            dx = minimumImage(x[i] - x)
            dy = minimumImage(y[i] - y)
            
            r2inv = 1.0/(dx**2 + dy**2 + tiny)
            c = 48.0 * r2inv**7 - 24.0 * r2inv**4
            fx = dx * c
            fy = dy * c
            
            fx[i] = 0.0 # no self force
            fy[i] = 0.0
            f[2*i] = fx.sum()
            f[2*i+1] = fy.sum()
            
            virial += np.dot(fx, dx) + np.dot(fy, dy)
                        
        return f, virial


    def lennardJonesForce_jit(self):
        N = self.N
        L = self.L
        
        x = self.x[::2]
        y = self.x[1::2]   
        f, virial = jit_lennardJonesForce(N, L, x, y)
                        
        return f, 0.5*virial
    
    def powerLawForce_jit(self):
        N = self.N
        L = self.L
        
        x = self.x[::2]
        y = self.x[1::2]   
        f, virial = jit_PowerLawForce(N, L, x, y)
                        
        return f, 0.5*virial


    def powerLawForce(self): 

        N = self.N
        virial = 0.0
        tiny = 1.0e-40 # prevents division by zero in calculation of self-force
        halfL = 0.5 * self.L
        
        x = self.x[::2]
        y = self.x[1::2]   
        f = np.zeros(2*N)
        minimumImage = self.minimumImage
        for i in range(N):  # The O(N**2) calculation that slows everything down
    
            dx = minimumImage(x[i] - x)
            dy = minimumImage(y[i] - y)
            
            r2 = dx**2 + dy**2 + tiny
            r6inv = pow(r2, -3)
            fx = dx * r6inv
            fy = dy * r6inv
            
            fx[i] = 0.0 # no self force
            fy[i] = 0.0
            f[2*i] = fx.sum()
            f[2*i+1] = fy.sum()
            
            virial += dot(fx, dx) + dot(fy, dy) 
                
        return f, 0.5 * virial 
            

    # TIME EVOLUTION METHODS 

    def verletStep(self): # Gould Eqs. 8.4a and 8.4b
    
        a = self.force()
        self.x += self.v * self.dt + 0.5 * self.dt**2 * a
        # collect unwraped x here for rms 

        self.x = self.x % self.L        # periodic boundary conditions
        self.v += 0.5 * self.dt * (a + self.force())
                            
            
    def evolve(self, time=10.0):

        steps = int(abs(time/self.dt))
        
        # When looping and adding values onto long lists, it is much
        # faster to use native python lists and append method, than 
        # numpy arrays
        self.tArray = self.tArray.tolist()
        self.temperatureArray = self.temperatureArray.tolist()
        self.EnergyArray = self.EnergyArray.tolist()        
        self.sampleTimeArray = self.sampleTimeArray.tolist()
        self.xArray = self.xArray.tolist()
        
        for i in range(steps):
        
            self.verletStep()
            # self.zeroTotalMomentum()
            
            self.t += self.dt
            self.tArray.append(self.t)
            
            if (i % self.sampleInterval == 0): # only calculate energy every sampleInterval steps to reduce load
            
                self.EnergyArray.append(self.energy())
                self.sampleTimeArray.append(self.t)
                self.xArray.append(self.x)
                # easier to keep track of structure of v array using np.append
                # it is only one np.append operation so does not slow things down
                self.vArray =  np.append(self.vArray, self.v)
            
            T = self.temperature()
            self.steps += 1
            self.temperatureArray.append(T)
            self.temperatureAccumulator += T
            self.squareTemperatureAccumulator += T*T

        # put everything back into numpy arrays 
        self.tArray = np.array(self.tArray)
        self.temperatureArray = np.array(self.temperatureArray)
        self.EnergyArray = np.array(self.EnergyArray)        
        self.sampleTimeArray = np.array(self.sampleTimeArray)
        self.xArray = np.array(self.xArray)
                    
                    
    def zeroTotalMomentum(self):
        # set total momentum to zero
        vx = self.v[::2]
        vy = self.v[1::2]
        
        vx -= vx.mean()
        vy -= vy.mean()
        
        vx = self.v[::2] = vx
        vy = self.v[1::2] = vy

                    
    def reverseTime(self):
    
        self.dt = -self.dt
                    
                    
    def cool(self, time=1.0):
    # artificially cool by rescaling velocities

        steps = int(time/self.dt)
        for i in range(steps):
                self.verletStep()
                self.v *= (1.0 - self.dt) # friction slows down atoms
                
        self.resetStatistics()

                    

# INITIAL CONDITION METHODS             
        
    def randomPositions_nocool(self, box_size = 1.):
        self.x = self.L * box_size * np.random.random(2*self.N)
        
    def randomPositions(self):

        self.x = self.L * np.random.random(2*self.N)
        
        self.forceType = "powerLaw_jit" 
        self.cool(time=0.1)
        self.forceType = "lennardJones_jit"


    def triangularLatticePositions(self):
        # if system is cooled from random positions it will naturally fall into a 
        # triangular lattice
        
        #self.rectangularLatticePositions()
        self.randomPositions()
        self.v += np.random.random(2*self.N) - 0.5 # jiggle to break symmetry
        
        self.forceType = "powerLaw_jit" 
        self.cool(time=10.0)
        self.forceType = "lennardJones_jit"


    def rectangularLatticePositions(self): # assume that N is a square integer (4, 16, 64, ...)
   
        if np.abs(np.sqrt(self.N) - np.floor(np.sqrt(self.N))) > 1e-10:
            N_new = np.rint(np.sqrt(self.N))**2
            self.N = int(N_new)
            self.x = np.zeros(2*self.N) # NumPy array of N (x, y) positions
            self.v = np.zeros(2*self.N) # array of N (vx, vy) velocities
            
            print("N must be a square integer to use this method\n"+
                    "Setting N to {} and reseting x and v".format(N_new))

        nx = int(np.sqrt(self.N))
        ny = nx
        dx = self.L / nx
        dy = self.L / ny
        
        for i in range(nx):
            x = (i + 0.5) * dx
            for j in range(ny):
                y = (j + 0.5) * dy
                self.x[2*(i*ny+j)] = x
                self.x[2*(i*ny+j)+1] = y

    def smallBoxPositions(self, box_L): # assume that N is a square integer (4, 16, 64, ...)
        # initialize particle positions in a small subregion of large box
    
        if np.abs(np.sqrt(self.N) - np.floor(np.sqrt(self.N))) > 1e-10:
            N_new = np.rint(np.sqrt(self.N))**2
            self.N = int(N_new)
            self.x = np.zeros(2*self.N) # NumPy array of N (x, y) positions
            self.v = np.zeros(2*self.N) # array of N (vx, vy) velocities
            print("N must be a square integer to use this method\n"+
                    "Setting N to {} and reseting x and v".format(N_new))

        nx = int(np.sqrt(self.N))
        ny = nx
        dx = box_L / nx
        dy = box_L / ny
        center = self.L/2 - box_L/2
        
        for i in range(nx):
            x = (i + 0.5) * dx
            for j in range(ny):
                y = (j + 0.5) * dy
                self.x[2*(i*ny+j)] = x + center
                self.x[2*(i*ny+j)+1] = y + center

    def randomVelocities(self):
    
        self.v = np.random.random(2*self.N) - 0.5

        # self.zeroTotalMomentum()
        
        T = self.temperature()

        # rescale to approximate initial temperature
        self.v *= np.sqrt(self.initialTemperature/T)
            
            
            
# MEASUREMENT METHODS

    def kineticEnergy(self):
    
        return 0.5 * (self.v * self.v).sum()
            
    
    def potentialEnergy(self):

        return self.lennardJonesPotentialEnergy()
            
            
    def lennardJonesPotentialEnergy(self): 
        N = self.N
        L = self.L
        
        x = self.x[::2]
        y = self.x[1::2]   
        U  = jit_lennardJonesPotentialEnergy(N, L, x, y)

        return U
    
    
    # def lennardJonesPotentialEnergy(self): # Gould Eqs. 8.1 and 8.2

        # tiny = 1.0e-40 # prevents division by zero in calculation of self-force
        # halfL = 0.5 * self.L
        # N = self.N
        
        # x = self.x[arange(0, 2*N, 2)]
        # y = self.x[arange(1, 2*N, 2)]   
        # U = 0.0
        # minimumImage = self.minimumImage
        # for i in range(N):  # The O(N**2) calculation that slows everything down
    
            # dx = minimumImage(x[i] - x)
            # dy = minimumImage(y[i] - y)

            # r2inv = 1.0/(dx**2 + dy**2 + tiny)
            # dU = r2inv**6 - r2inv**3
            # dU[i] = 0.0 # no self-interaction
            # U += dU.sum()

        # return 2.0 * U
            
    def energy(self):
    
        return self.potentialEnergy() + self.kineticEnergy()
            
    def temperature(self): # Gould Eq. 8.6
    
        return self.kineticEnergy() / self.N

# STATISTICS METHODS            
    def resetStatistics(self):
    
        self.steps = 0
        self.temperatureAccumulator = 0.0
        self.squareTemperatureAccumulator = 0.0
        self.virialAccumulator = 0.0
        self.xArray = np.array([])
        self.vArray = np.array([])

    def meanTemperature(self):
    
        return self.temperatureAccumulator / self.steps
            
            
    def meanSquareTemperature(self):
            
        return self.squareTemperatureAccumulator / self.steps
            
            
    def meanPressure(self): # Gould Eq. 8.9
    
        meanVirial = 0.5 * self.virialAccumulator / self.steps # divide by 2 because force is calculated twice per step
        return 1.0 + 0.5 * meanVirial / (self.N * self.meanTemperature())
            
            
    def heatCapacity(self): # Gould Eq. 8.12

        meanTemperature = self.meanTemperature()
        meanSquareTemperature = self.meanSquareTemperature()
        sigma2 = meanSquareTemperature - meanTemperature**2
        denom = 1.0 - sigma2 * self.N / meanTemperature**2
        return self.N / denom

    def meanEnergy(self):
    
        return self.EnergyArray.mean()
            
    def stdEnergy(self):
    
        return self.EnergyArray.std()
            
            
# PLOTTING METHODS
                            
    def plotPositions(self):
    
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(self.x[::2], self.x[1::2], s=50.0, marker='o', alpha=1.0)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
            
            
    def plotTrajectories(self, number=1):
        # number is the number of trajectories to plot 
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        N = self.N
        size = np.size(self.xArray)//(2*N)
        r = np.reshape(self.xArray, [size, 2*N])
        for i in range(number):
                x = r[:, 2*i]
                y = r[:, 2*i+1] 
                ax.plot(x, y, ".")
        
            
    def plotTemperature(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.plot(self.tArray, self.temperatureArray)
        ax.set_xlabel("time")
        ax.set_ylabel("temperature")
            
            
    def plotEnergy(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        Emin = np.min(self.EnergyArray)
        Emax = np.max(self.EnergyArray)
        
        ax.plot(self.sampleTimeArray, self.EnergyArray)
        ax.set_ylim = [5*Emin, 5*Emax]
        ax.set_xlabel("time")
        ax.set_ylabel("Energy")
            
            
    def velocityHistogram(self):
        vx = self.vArray[::2]
        vy = self.vArray[1::2]
        plt.figure(5)
        plt.hist(self.vArray, bins= 100, normed=1)
        plt.xlabel("velocity in x- or y-directions")
        plt.ylabel("probability")

        v = np.sqrt(vx**2+vy**2)
        plt.figure(6)
        plt.hist(v, bins= 'auto', normed=1)
        plt.xlabel("speed")
        plt.ylabel("probability")
            
            
    def showPlots(self):
            plt.show()

    def reverseTime(self):
    
        self.dt = -self.dt
                    
                    
    def cool(self, time=1.0):

        steps = int(time/self.dt)
        for i in range(steps):
                self.verletStep()
                self.v *= (1.0 - self.dt) # friction slows down atoms
                
        self.resetStatistics()

# RESULTS METHODS
    def results(self):
        print("\n\nRESULTS\n") 
        print("time = ", self.t, " total energy = ",self.energy(), " and temperature = ", self.temperature())
        if (self.steps > 0):
            print("Mean energy = ", self.meanEnergy(), " and standard deviation = ", self.stdEnergy())
            print("Cv = ", self.heatCapacity(), " and PV/NkT = ",self.meanPressure())


######################################
#####################################


# IDEAL GAS PROPERTIES.
def ideal_gas(Temp = 100):
    gas = MolecularDynamics(N=64, L=16, initialTemperature = Temp, dt = 0.0001, forceType = "lennardJones_jit") 
    gas.rectangularLatticePositions()
    
    # gas.triangular
    gas.triangularLatticePositions()
    gas.randomVelocities()
    gas.plotPositions()
    gas.results()

    gas.evolve(time=1.0) # initial time evolution
    gas.resetStatistics() # remove transient behavior
    gas.evolve(time=10.0) # accumulate statistics 
    gas.results()

    gas.velocityHistogram()
    gas.plotEnergy()
    gas.plotTemperature()
    gas.plotTrajectories(gas.N/8)
    gas.showPlots()

    return gas


# TIME-REVERSAL TEST
def time_reverse(N = 16, t_max=1, dt = 10):
       
    # method to enable animation pause on mouse click
    anim_running = True
    def onClick(event):
        nonlocal anim_running
        if anim_running:
            ani.event_source.stop()
            anim_running = False
        else :
            ani.event_source.start()
            anim_running = True
    
    # method to update figure in animation
    def updatefig(i, t_max, dt, cmap):
        
        if md.dt > 0:
            time_direction = "Forward Time\n"

        elif md.dt < 0:
            time_direction = "Reverse Time\n"
        
        if (md.t > t_max and md.dt > 0):
            # print('Reverse Time')
            md.reverseTime()
            # ani.event_source.stop()
            # anim_running = False
        
        if (md.t < 1*md.dt and md.dt < 0):
            # print('Reverse Time')
            md.reverseTime()
            # ani.event_source.stop()
            # anim_running = False

        md.resetStatistics()
        md.evolve(time = np.abs(md.dt*dt))

        # update the data in plot during animation
        xy = np.reshape(md.x, [2, md.N], order = 'F')
        trajectories.set_offsets(xy.T)
        # trajectories.set_edgecolors(color)
        trajectories.set_facecolors("C0")
        trajectories.set_alpha(1)
        tracker.set_data(md.x[point], md.x[point+1])
        tracker.set_alpha(0.8)
        time_text.set_text(time_direction + 't = {:0.4f}'.format(md.t))
        
        # every object that is updated at each animation frame must
        # be returned by this function
        return trajectories, tracker, time_text
    
    # instantiate object
    md = MolecularDynamics(N=N, L=25, initialTemperature = 10, dt = 0.001, forceType = "lennardJones_jit") 
    # Put all particles at center of large box
    md.smallBoxPositions(box_L=8)
    md.randomVelocities()

    cmap = cm.get_cmap('coolwarm')
    rgba = cmap(0)
    fig = plt.figure(figsize = [8,8])
    ax = fig.add_subplot(111)
    ax.set_ylabel('y', fontsize = 16)
    ax.set_xlabel('x', fontsize = 16)
    
    # pick some point for our tracker particle
    point = int(md.N - np.sqrt(md.N))
    
    # enable plot interaction
    fig.canvas.mpl_connect('button_press_event', onClick)

    # Plot trajectories
    trajectories = ax.scatter(md.x[::2], md.x[1::2],s = 150, marker='o', edgecolors = 'C0', facecolors = 'None', alpha = 0.9)

    # Plot tracker particle
    tracker = ax.plot(md.x[point], md.x[point+1], marker = 'o', ms = 10,color = 'C1', alpha = 0.7)[0]
    time_text = ax.text(0.05, 0.90, '', transform=ax.transAxes, fontsize = 16)
    
    # generate animation
    ani = am.FuncAnimation(fig, updatefig, fargs = [t_max, dt, cmap], interval = 10, frames = 1000, blit = True)
    plt.xlim(0,1.05*md.L)
    plt.ylim(0,1.05*md.L)
    
    plt.show()

    return md


def liquidsolid():

    # instantiate object
    md = MolecularDynamics(N=64, L=8, initialTemperature = 0.001, dt = 0.0001, forceType = "lennardJones_jit") 
    md.rectangularLatticePositions()
    md.randomVelocities()
    md.plotPositions()
    ax = plt.gca()
    ax.set_title('t = 0')
    
    
    md.evolve(time=0.7) # initial time evolution
    md.plotTrajectories(md.N)
    ax = plt.gca()
    ax.set_title('t = 0 $\\rightarrow$ 0.7')

    md.evolve(time=0.1) 
    md.resetStatistics() # remove transient behavior
    md.evolve(time=4.2) # second time evolution
    md.plotTrajectories(md.N)
    ax = plt.gca()
    ax.set_title('t = 0.8 $\\rightarrow$ 5')

    md.evolve(time=11)     
    md.resetStatistics() # remove transient behavior
    md.evolve(time = 6) # final time evolution
    md.plotTrajectories(md.N)
    ax = plt.gca()
    ax.set_title('t = 16 $\\rightarrow$ 22')

    md.plotTemperature()

    md.showPlots()


# simulate freezing by continuously cooling gas
def freezing(Tstart = 500, N = 400, L = 20, dt = 1e-4):
   
    anim_running = True
    def onClick(event):
        nonlocal anim_running
        if anim_running:
            ani.event_source.stop()
            anim_running = False
        else :
            ani.event_source.start()
            anim_running = True
 
    # instantiate object
    md = MolecularDynamics(N=N, L=L, initialTemperature=Tstart, dt = dt, forceType = "lennardJones_jit") 

    # this method is require for the animation
    def updatefig(i,Tmax, cmap):
        md.resetStatistics()
        md.evolve(time = md.dt*10)
        
        # set color based on average temperature
        T = md.temperature()

        if T < 0.5:
            md.reverseTime()
        if T > Tmax*1.5:
            md.reverseTime()
        # normalization for colorscale
        norm = colors.LogNorm(vmin = 0.5, vmax=Tmax)
        color = cmap(norm(T))
       
        # update the data in plot during animation
        xy = np.reshape(md.x, [2, md.N], order = 'F')
        trajectories.set_offsets(xy.T)
        trajectories.set_edgecolors(color)
        trajectories.set_facecolors(color)
        md.cool(md.dt*50)
        T_text.set_text('Temperature = {:0.3f}'.format(md.temperature()))

        return trajectories, T_text
    
    # colormap used to represent temperature
    cmap = cm.get_cmap('coolwarm')
   
    fig = plt.figure(figsize = [8, 8])
    ax = fig.add_subplot(111)
    ax.set_ylabel('y', fontsize = 16)
    ax.set_xlabel('x', fontsize = 16)
    ax.axis('off')
    ax.patch.set_visible(False) 
    ax.set_aspect('equal', 'box')
    fig.patch.set_visible(False) 
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    fig.canvas.mpl_connect('button_press_event', onClick)
    

    # norm = Normalize(vmin = 0.001, vmax = 1)
    # mappable = cm.ScalarMappable(norm = norm , cmap = cmap)
    # mappable.set_array([])
    
    # cax = fig.add_axes([0.10, 0.90, 1.8,0.1])
    # cb = fig.colorbar(mappable, cax, cmap = cmap, norm=norm, orientation = 'horizontal', drawedges = False, ticklocation = 'bottom')
    # cb.solids.set_edgecolor('face')
    # cb.set_label('Temperature', rotation = 0, labelpad = 12)
    # cb.set_ticks(MultipleLocator(climit[1]/2))
    # cb.formatter.set_powerlimits((-2,2))
    # cb.update_ticks()


    # set initial conditions 
    md.rectangularLatticePositions()
    # md.triangularLatticePositions()
    md.randomVelocities()
    T_max = md.temperature()
    rgba = cmap(1.0)

    # initial plot object is just a placeholder for animation
    trajectories = ax.scatter([], [],s = 100, marker='o', edgecolors = rgba, facecolors = rgba)
    T_text = ax.text(0.05, 0.92, ' ', transform=ax.transAxes, color = 'w', fontsize = 16)
    
    # create animation object
    ani = am.FuncAnimation(fig, updatefig, fargs = [T_max,  cmap], interval = 10, frames = 1000, blit = True)
    plt.xlim(0,1.0*md.L)
    plt.ylim(0,1.1*md.L)
    plt.show()

    return md, ani 
