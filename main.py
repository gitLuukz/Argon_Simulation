"""
Main document of the project
"""

"""
Importing of libraries which are needed in this file
"""
# Import libraries
import numpy as np

# Import py files with written functions
import plotfunctions
import particles_verlet as pv

"""
Initialisation of the project
Generate parameters and initial positions, forces and velocity
"""

# Parameters
epsilon  = 1#119.8 # normalised with boltzmann constant
sigma = 1#3.405e-10 

#Density of the system
rho = 0.88
const = 2/3*(14/rho)**(1/3) 
print('found lattice constant')
print(const)
print('set density')
print(rho)
D = 3*const # Dimension of system 

#values for plot windows
area = np.array([-0.1, 1.1, -0.1, 1.1])*D
area_anim = np.array([0, 1, 0, 1])*D

#(reduced) constants of the system
m = 1 # mass of argon atom
dim = 3 # Spatial Dimension of the problem
kb = 1 #Boltzman constant
T = 0.88 #1/119.8 # Desired temperature of the system in Kelvin/kb

# Particle Initialisation
init = pv.FCC(D,const)
N = int(init[1]) # number of particles 
print('calculated density using whole volume and total particles')
print(N/D**3)
Rv = [init[0]] # for Verlet method
plotfunctions.start_postion(1, Rv, area, D)

# Time evolution of the problem
n = 50*10 # steps
a = 2*1 # End time of plot
t = np.linspace(0,a,n) # time
dt = a/(n) # timestep
n0 = int(n/2)
t0 = np.linspace(a/2,a,int(n0)) #time for the observables


# Speed initialisation
V = np.zeros((3,N))
V = pv.particle_initial_velocity(6,N,D,T,m,dim,kb) 

""" 
Simulation with verlet evaluation 
"""
# Simulate movement with same starting position as Euler
steps_r = 60 #amound of bins in pair correlation
[Rv,Ekinv,Epotv,Ev,Gpc] = pv.simulating_verlet(n, N, D, t, Rv, sigma, epsilon, dt, m,T,dim,kb,V,steps_r)

# Calculation of specific heat and error
Cv = pv.specific_heat(Ekinv, n, N,t0)
Cv_error = pv.error_specific_heat(Cv)

#Pair correlation calculated and plotted
errorPC = pv.error_pc(Gpc, steps_r, n0, t)
pc = pv.pair_correlation(N,Rv[-1],D,steps_r)
plotfunctions.pair_correlation(8,*pc,errorPC)

#Calculating error kinetic energy
E_error = pv.error_Energy(t0,Ekinv,n0)

# Plotting the energy of the system
plotfunctions.Energies(2, t, Ekinv, Epotv, Ev ,'Verlet')

# Plotting the total energy variation
plotfunctions.variation_total_energy(3, t, Ev, 'Verlet')

# Animation 
ANIMATE =False #toggle if you want to make a gif or not
plotfunctions.animation_movement(5, ANIMATE, Rv, area_anim, D, n,'Verlet')