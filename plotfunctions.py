"""
File for functions that can plot results of files like project_1*.py
"""

# Import libraries for plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from scipy.stats import norm
import numpy as np

def start_postion(fignr,R,area,D):
    # Figure of start position
    fig1 = plt.figure(fignr, figsize = (5,5))
    ax1 = fig1.add_subplot(111, projection = '3d')
    ax1.scatter(R[0][0,:], R[0][1,:], R[0][2,:])
    plt.axis(area)
    ax1.set_zlim3d(-0.1*D,1.1*D)
    plt.title('Starting position of the particles')
    plt.xlabel('x [a.u.]')
    plt.ylabel('y [a.u.]')
    ax1.set_zlabel('z [a.u.]')
    plt.savefig('../start_position_3D.png')
    return

def velocity(fignr,N,V):
    plt.figure(fignr, figsize = (5,5))
    n,bins,patches = plt.hist(V[1,:],30)
    mu,sigma = norm.fit(V[1,:])
    g = np.exp(-1/2*((bins-mu)/sigma)**2)*1/sigma/np.sqrt(2*np.pi)
    g = g/np.max(g)*np.max(n)
    plt.plot(bins,g,label = 'fit of start velocity')
    # print('Fit values for start velocity')
    # print('Mu')
    # print(mu)
    # print('Sigma')
    # print(sigma)
    plt.title('The initial velocity')
    plt.legend()
    return

def correlation(fignr,g,N):
    plt.figure(fignr, figsize = (5,5))
    plt.hist(g,N)
    plt.title('The correlation function')
    return

def Energies(fignr,t,Ekin,Epot,E,name):
    plt.figure(fignr, figsize = (5,5))
    plt.plot(t[1:],Ekin[1:],label = 'Kin')
    plt.plot(t[1:],Epot[1:],label = 'Pot')
    plt.plot(t[1:],E[1:], label = 'E')
    plt.legend()
    plt.title('The variation of the kinetic, potential and total energy')
    plt.xlabel('Time [a.u.]')
    plt.ylabel('Energy [a.u.]')
    plt.savefig('../Energies_'+str(name)+'.png')
    return

def Energy_Euler_Verlet(fignr,t,Ev,E):
    plt.figure(fignr, figsize = (5,5))
    plt.plot(t[1:],Ev[1:],label = 'Verlet')
    plt.plot(t[1:],E[1:],label = 'Euler')
    plt.legend()
    plt.title('The variation of the total energy')
    plt.xlabel('Time [a.u.]')
    plt.ylabel('Energy [a.u.]')
    plt.savefig('../Total_energy_comparison.png')
    return

def variation_total_energy(fignr,t,Energy,name):
    plt.figure(fignr, figsize = (5,5))
    plt.plot(t[1:],Energy[1:])
    plt.title('The variation of the total energy')
    plt.xlabel('Time [a.u.]')
    plt.ylabel('Energy [a.u.]')
    plt.savefig('../Total_energy_variation'+str(name)+'.png')
    return

def distance_two_part(fignr,inter_dis,t,name):
    plt.figure(fignr, figsize = (5,5))
    plt.plot(t[1:],inter_dis[1:])
    plt.title('The distance between the particles')
    plt.xlabel('Time [a.u.]')
    plt.ylabel('Distance [a.u.]')
    plt.savefig('../Distance_particels_'+str(name)+'.png')
    return

def movement_particles(fignr,R,t,D,area,name):
    # Figure of movement
    fig = plt.figure(fignr, figsize = (5,5))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(1,len(t)):
        ax.scatter(R[i][0,:], R[i][1,:], R[i][2,:])
        
        
    plt.axis(area)
    ax.set_zlim3d(-0.1*D,1.1*D)   
    plt.title('The trajectory of the particles')
    plt.xlabel('x [a.u.]')
    plt.ylabel('y [a.u.]')
    ax.set_zlabel('z [a.u.]')
    plt.savefig('../end_position_3D_'+str(name)+'.png') 
    return

def animation_movement(fignr,ANIMATE,R,area_anim,D,n,name):
    """ 
    Function that will make a animation of the movement
    It will save a gif one level above the working directory
    fignr depens on 
    """
    if ANIMATE == True:
        def update_plot(i):
            graph._offsets3d = (R[i][0,:], R[i][1,:], R[i][2,:])
        
        fig2 = plt.figure(fignr,figsize = (5,5))
        ax2 = fig2.add_subplot(111, projection = '3d')
        plt.axis(area_anim)
        ax2.set_zlim3d(0,D)
        graph = ax2.scatter(R[0][0,:], R[0][1,:], R[0][2,:])
        plt.title('The animation of the particles')
        plt.xlabel('x [a.u.]')
        plt.ylabel('y [a.u.]')
        ax2.set_zlabel('z [a.u.]')
                  
        animationgif = animation.FuncAnimation(fig2, update_plot, n, interval = 10, blit = False)
        plt.close()
        
        """ Saving the animation gives a warning but does work if enabled"""
        animationgif.save('../animation_of_particles_3D_'+str(name)+'.gif')
    return

def pair_correlation(fignr,g,dist,dr,errorPC):
    plt.figure(fignr,figsize = (5,5))
    plt.bar(dist,g,dr,yerr = errorPC,align = 'center', capsize =3, alpha = 0.9, ecolor = 'k')
    plt.title('Pair correlation')
    plt.xlabel('distance r')
    plt.ylabel('g(r)')
    plt.savefig('../paircorrelation.png') 
    plt.ylim((0,max(g)*(1.2)))
    return