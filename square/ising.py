import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from matplotlib import colors


class Ising:
    ''' A class that implements the Ising model
    L: size of square lattice
    J: interaction energy
    B: strength of external magnetic field
    beta: 1/kT Units of 1 and defaults to 1
    
    '''
    def __init__(self, L,J,B, beta=1) -> None:
        self.L = L
        self.J = J
        self.B = B
        self.T = 273.15 # K
        self.k = 1.381e-23  # m2 kg s-2 K-1
        self.beta = beta #/ (self.k*self.T)
        self.lattice = 2*np.random.randint(2, size=(L,L)) - 1 #spins either -1 or 1
        self.visualisations = []
        self.variance = 0
    
    
    def interaction_energy(self):
        return np.sum(self.lattice, self.lattice) * -self.J
    
    def magnetic_energy(self):
        return np.sum(self.lattice, -self.B)
    
    def total_energy(self):
        return self.interaction_energy() + self.magnetic_energy()
    
    def deltaE(self,i,j):
        f = (self.lattice[(i+1)%self.L,j] + self.lattice[(i-1)%self.L,j] + self.lattice[i,(j+1)%self.L] + self.lattice[i,(j-1)%self.L])
        
        return -2 * (self.J * f + self.B) * self.lattice[i,j]
    
    
    def flip_spin(self, i, j):
        self.lattice[i,j] *= -1
        
    def should_flip(self, i,j) -> bool:
        dE = self.deltaE(i,j)
        rand_num = np.random.rand()
        prob = np.exp(-dE/self.beta)
        return prob > 1 or prob > rand_num
    
    def update_magnetic_field(self, b):
        self.B = b
    
    def do_thermalisation_sweep(self, n = 1):
        for k in range(n):
            for i in range(self.L):
                for j in range(self.L):
                    self.flip_spin(i,j)
                    if not self.should_flip(i,j):
                        # flip back if false
                        self.flip_spin(i,j)
                    
    def get_mag_per_spin(self):
        return np.sum(self.lattice) / self.L**2
    
    def variance(self):
        self.variance = np.var(self.lattice)
    
    def get_energy_per_spin(self):
        return self.total_energy() / (2 * self.L**2)
    
    def get_succeptibility(self):
        return self.beta * np.var(self.lattice) / self.L**2
    
    def get_specific_heat(self):
        return 1/4 * self.beta**2 * np.var(self.lattice)  / self.L**2
    
    
    
    def visualise(self, i):
        fig, ax = plt.subplots()  # Create a new figure and axes
        cmap = colors.ListedColormap(['#482878','#DBE319'])
        bounds = [-1, 0, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        ax.imshow(self.lattice, vmin=-1, vmax=1, cmap=cmap)
        # ax.set_title(f'Thermalisation Sweep: {i}')
        ax.axis('off')
        fig.savefig(f'images/ising_{i}.png', dpi=300)
        plt.close(fig)  # Close the figure to avoid displaying it here
        self.visualisations.append(fig)  # Save the figure in the list
        

    
