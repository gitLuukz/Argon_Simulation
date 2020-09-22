"""
Function file to compute particle movement
Time steps are simulated with Verlet method
"""

# libraries Needed
import numpy as np
import copy
import plotfunctions
from scipy.optimize import curve_fit

# Functions
def FCC(D,const):
    """
    Parameters
    ----------
    D : Length of the system in x,y and z direction
    const: Lattice constant of argon
    Returns
    -------
    R : Starting positions of the particles
    N : Amount of particles in the system depending on the dimension D
    """
    a = np.array([[0,0,1/2,1,1],[0,1,1/2,0,1],[0,0,0,0,0]])
    b = np.array([[0,1/2,1/2,1],[1/2,0,1,1/2],[1/2,1/2,1/2,1/2]])
    c = a+np.array([[0,0,0,0,0],[0,0,0,0,0],[1,1,1,1,1]])
    blok = np.concatenate((a,b,c),1)
    R = np.array([[],[],[]])
    R = np.concatenate((R,blok),1)*const
    x = np.array([[1],[0],[0]])*const
    y = np.array([[0],[1],[0]])*const
    z = np.array([[0],[0],[1]])*const
    R = R+1/4*const
    for i in range(int(D/const)):
        R = np.concatenate((R,R+x),1)
        R = np.concatenate((R,R+y),1)
        R = np.concatenate((R,R+z),1)
    R = np.unique(R,axis = 1)
    lok = np.argwhere(R>=D)
    lok = lok[:,1]
    lok = np.unique(lok)
    R = np.delete(R,lok,1)
    N = R.size/3
    return R,N
    
def particle_initial_velocity(fignr,N,D,T,m,dim,kb):
    """
    V is the matrix with the initial velocities in x,y,z direction which have been put to zero.
    Input
        fignr: number for figure in which to plot
        N: Amount of particles
        D: Dimension of the system
        T: temperature
        m: mass 
        dim: spatial dimensions
        kb: boltzmann constant
    output
        V: Velocity matrix (3xN) containing x,y and z velocity for all particles
    """
    V = np.zeros((3,N))
    V[0:dim,:] = np.random.normal(0, kb*T/m, (dim,N))# / np.sqrt(T/(kb*m))
    plotfunctions.velocity(fignr,N,V)
    # Typical speed for particles
    return V

def check_bc(R,D,p,mn):
    """
    Check if the particle moves through the boundary. If that is the case move the particle to the other side of the box.
    input:
        R: position matrix (3xN)
        D: dimension length of the system
        p: upper limit for check
        mn: lower limit for check
    output: 
        R: New position matrix where particles will be inside the system dimensions
    """
    R -= (R/D>=p)*np.floor(R/D)*D
    R -= (R/D<=mn)*np.floor(R/D)*D
    return R

def particle_positionV(R,V,dt,F,D):
    """
    This function returns the new position after interaction with other particles.
    
    input:
        R: Position matrix (3xN)
        V: velocity matrix (3xN)
        dt: time step size
        F: Force matrix (3xN)
        D: system length
    output:
        R: new Position matrix after time dt
    """
    R += (dt*V+dt**2/2*F) 
    p = 1
    mn = 0
    R = check_bc(R, D, p, mn)
    return R

def minimal_image(x,y,z,D):
    """"
    This function checks if mirror images are closer than the particles in the box.
    
    input:
        x: vector containing all x distances to 1 particle
        y: vector containing all y distances to 1 particle
        z: vector containing all z distances to 1 particle
        D: length of the system
    output:
        x,y and z are new vectors but checked for minimal image convention
    """
    diff = D
    x = (x+diff/2) % D - diff/2
    y = (y+diff/2) % D - diff/2
    z = (z+diff/2) % D - diff/2
    return x, y, z

def particle_forceV(R,N,sigma,epsilon,D):
    """
    The F matrix contains the force acting on the particles in the x, y and z direction.
    Directly checks for the minimal image method
    
    input: 
        R: position matrix (3xN)
        N: Number of particles
        sigma: constant
        epsilon: constant
        D: length of the system
    
    output:
        F: Force matrix (3xN) containing the force on every particle in x,y and z direction
    """
    F = np.zeros((3,N))
    x = np.zeros(N-1)
    y = np.zeros(N-1)
    z = np.zeros(N-1)
    r = np.zeros(N-1)
    # loop over all particles
    for i in range(N):
        # Distances for x,y,z between particles
        x = R[0,np.arange(N)!=i]-R[0,i]
        y = R[1,np.arange(N)!=i]-R[1,i]
        z = R[2,np.arange(N)!=i]-R[2,i]
        [x,y,z] = minimal_image(x,y,z,D)
        c = np.stack((x,y,z))
        r = np.sqrt(np.sum(c**2,0))
        a = (c*4*(sigma/epsilon)*(12/r**14-6/r**8))
        F[:,i] = -np.sum(a,1)
    return F

def particle_velocityV(V,F,dt,Rv,sigma,epsilon,D,N):
    """
    V is the matrix with the velocities of the particles, in x,y,z. The V matrix is created using time evolution. 
    
    input:
        V: velocity matrix (3xN)
        F: Force matrix (3xN)
        dt: time step size
        Rv: list that contains position matrixes 
        sigma,epsilon: constants
        D: dimension length system
        N: number of particles
    Output:
        V: New velocity after time dt
    """       
    V += dt/2*(particle_forceV(Rv[-1], N, sigma, epsilon, D) + particle_forceV(Rv[-2], N, sigma, epsilon, D))
    return V

def calibration(N,kb,T,Ekinv,V):
    """  
    Calibrates the velocity if the kinetic energy variates to much from the assigned energy (by temperature)
    input:
        N: Number of particles
        kb: constant
        T: temperature
        Ekinv: Kinetic energy vector
        V: velocity
    output:
        V: Calibrated velocity
    """
    lamb = np.sqrt((N-1)*3*kb*T/(Ekinv*2))
    
    if lamb < 0.9999:
        V = lamb*V
    elif lamb>1.0001:
        V = lamb*V
    
    return V

def particle_LJV(R,N,D):    
    """
    Calculation of the Lennart Jones potential.
    
    Input: 
        R: position matrix
        N: number of particles
        D: system length
    output:
        Uv: The value of the potential 
    """
    b = np.zeros(N)
    for i in range(N):
        x = R[0,np.arange(N)!=i]-R[0,i]
        y = R[1,np.arange(N)!=i]-R[1,i]
        z = R[2,np.arange(N)!=i]-R[2,i]
        [x,y,z] = minimal_image(x,y,z,D)
        c = np.stack((x,y,z))
        r = np.sqrt(np.sum(c**2,0))
        b[i] = np.sum(4*((1/r)**12-(1/r)**6))
    Uv = np.sum(b)
    return Uv

def pair_correlation(N,R,D,steps_r):
    """ 
    Calculates the pair correlation depending on the positions of the particles (R).
    
    Input:
        N: number of particles
        R: position matrix
        D: system length
        steps_r: amound of bins
    
    Output:
        g: pair correlation
        dist: distances for which it is calculated (needed for bar plot)
        dr: distance between bars
    """
    rmax = D
    dist = np.linspace(0.001,rmax,steps_r)
    dr = rmax/steps_r
    n =  np.zeros((N,steps_r))
    for i in range(N):
        # Distances for x,y,z between particles
        x = R[0,np.arange(N)!=i]-R[0,i]
        y = R[1,np.arange(N)!=i]-R[1,i]
        z = R[2,np.arange(N)!=i]-R[2,i]
        [x,y,z] = minimal_image(x,y,z,D)
        c = np.stack((x,y,z))
        r = np.sqrt(np.sum(c**2,0))
        for j in range(steps_r):
            n[i,j] = np.sum((r<(dist[j]+dr)) * (dist[j]<r))
    n_avg = 1 / N * np.sum(n,0)
    g = 2 * D**3 / (N * (N - 1)) / (4 * np.pi * dist**2 * dr) * n_avg
    return g, dist, dr

def simulating_verlet(n,N,D,t,Rv,sigma,epsilon,dt,m,T,dim,kb,V,steps_r):
    """  
    Simulation of argon gas by iteration. System is stabilised by calibration of the velocities.
    
    Input:
        n: iteration steps
        N: number of particles
        D: system length
        t: time vector
        Rv: list containing position matrixes 
        sigma,epsilon: constant
        dt: time step size
        m: mass constant
        T: temperature
        dim: dimensions of the system
        kb: boltzmann constant
        V: velocity matrix
        steps_r: amound of bins used in pair correlation
        
    Output:
        Rv: list of all particle locations for every iteration
        Ekinv: Kinetic energy for every iteration
        Epotv: Potential energy for every iteration
        Ev: total energy for every iteration
        Gpc: The pair correlation for every iteration
    """
    Ekinv = np.zeros((n,1))
    Epotv = np.zeros((n,1))
    Ev = np.zeros((n,1))
    Gpc = np.zeros((steps_r,n))
    for k in range(len(t)):
        F = particle_forceV(Rv[-1], N, sigma, epsilon, D)
        Rv.append(particle_positionV(copy.deepcopy(Rv[-1]), V, dt, F, D)) 
        V = particle_velocityV(V, F, dt, Rv, sigma, epsilon, D, N)
        Ekinv[k] = np.sum(1/(2*m)*(V**2))
        
        #Calibration
        if (int(k%(10)) == int(0) & int(k)<int(len(t)/2)):
            V = calibration(N, kb,T,Ekinv[k],V)
            Ekinv[k] = np.sum(1/(2*m)*(V**2))
        if int(k)> int(len(t)-50):
            Gpc[:,k], dist, dr = pair_correlation(N,Rv[-1],D,steps_r)
        Uv = particle_LJV(Rv[-1], N, D) 
        Epotv[k] = abs(Uv)/2 
        Ev[k] = Ekinv[k]+Epotv[k]
    return Rv, Ekinv, Epotv, Ev, Gpc

def error_pc(Gpc,steps_r,n0,t0):
    """
    Input:
        Gpc: Pair correlation matrix for different time steps.
        steps_r:  amount of bins later used in plot.
        n0 :    iterations/2.
        t0 :    time for fitting.

    Returns:
        errorPC :    Error of every bin in pair correlation.
    """
    errorPC = np.zeros(steps_r)
    
    a = 20
    b = a+1
    t0 = np.linspace(0,a-1,a)
    for i in range(steps_r-1):
        chi2d= auto_correlation(t0,Gpc[i+1,-b:-1], a)#n0
        tau, popt = curve_fit(exp,t0,chi2d, p0 = 50)
        
        errorPC[i] = error(tau,a,Gpc[i+1,-b:-1],t0)
    return errorPC

def specific_heat(Ekinv,n0,N,t0):
    """
    Calculate the specific heat.
    input: 
        Ekinv: Kinetic energy value
        n0: value of steps in system
        N: amount of particles
        t0: time vector
    ouput:
        Cv: specific heat value 
    """
    Cv = np.zeros(10)
    for i in range(10):
        avg_K_squared = (1 / len(Ekinv[int(n0-50+5*i):int(n0-46+5*i)]) * np.sum(Ekinv[int(n0-50+5*i):int(n0-46+5*i)]))**2
        FluctK = 1 / len(Ekinv[int(n0-50+5*i):int(n0-46+5*i)]) * np.sum(Ekinv[int(n0-50+5*i):int(n0-46+5*i)]**2) - avg_K_squared
        Cv[i] = -1 / (FluctK/ avg_K_squared * 3 * N / 2 - 1) * 3 * N / 2
    return Cv


def error_specific_heat(Cv):
    """
    Calculates the error of the specific heat
    input:
        Cv: specific heat
    output: 
        error_Cv: the error for the value of Cv
    """
    ncv = 10
    tcv = np.linspace(0,50,ncv) 
    
    chi = auto_correlation(tcv,Cv,ncv)
    tau_fit = curve_fit(exp,tcv,chi)
    tau = tau_fit[0]

    error_Cv = error(tau,ncv,Cv,tcv) 

    return error_Cv

def error_Energy(t0,E,n0):
    """
    input
        t0 : Time vector
        E : Energy vector
        n0 : Amount of steps

    output:
        error_E : Error in the Energy E
    """
    chi = auto_correlation(t0,E,n0)
    tau_fit = curve_fit(exp,t0,chi)
    tau = tau_fit[0]

    error_E = error(tau,n0,E,t0)
    return error_E
   
def auto_correlation(t,A,N):
    """ 
    Calculates the Autocorrelation
    Input:
        t: time vector
        A: input matrix
        N: number of steps
    ouput:
        chi: autocorrelation of A
    """
    chi = np.zeros(len(t))
    for i in range(len(t)):
        chi[i] = 1/(N-i)*np.sum(A[0:N-i]*A[i:N]) - 1/(N-i)*np.sum(A[0:N-i])*1/(N-i)*np.sum(A[i:N])   
    return chi

def exp(t,tau):
    """ 
    Exponential function used to fit the autocorrelation. Can be used to obtain a tau value
    
    t: time vector
    tau: to be found value when fitting
    """
    return np.exp(-t/tau)

def error(tau,n0,A,t0):
    """
    General error function
    input:
        tau: 
        n0: number of steps
        A: input vector 
        t0: time vector
    output: 
        error: error of input value A
    """
   
    error = np.sqrt( 2*tau/(n0) *abs(np.sum(A**2)/len(A) - (np.sum(A)/len(A)**2 )))
    return error
