import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display


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
        
    def total_energy(self):
        return np.dot(self.lattice, self.B)
    
    def interaction_energy(self):
        return np.dot(self.lattice, self.lattice) * -self.J
    
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
    
    def do_thermalisation_sweep(self):
        for i in range(self.L):
            for j in range(self.L):
                self.flip_spin(i,j)
                if not self.should_flip(i,j):
                    # flip back if false
                    self.flip_spin(i,j)
        
        mag_per_spin = np.sum(self.lattice) / self.L**2
        
        return mag_per_spin
    
    def visualise(self, i):
        fig, ax = plt.subplots()  # Create a new figure and axes
        ax.imshow(self.lattice, vmin=-1, vmax=1)
        ax.set_title(f'Thermalisation Sweep: {i}')
        self.visualisations.append(fig)  # Save the figure in the list
        plt.close(fig)  # Close the figure to avoid displaying it here
        



L, B, J, beta = 32, -0.05, 0.5, 30
ising = Ising(L,J,B, beta)

num_therm = 30


mag_per_spin_arr = np.zeros(num_therm)
visual_arr = []

for i in range(num_therm):
    mag_per_spin = ising.do_thermalisation_sweep()
    mag_per_spin_arr[i] = mag_per_spin
    ising.visualise(i+1)


# plot magnetisation per spin array against numper of thermalisation sweeps
plt.plot(mag_per_spin_arr)


print(mag_per_spin_arr[-1])

# Create a slider to select the thermalisation sweep to display
slider = widgets.IntSlider(min=0, max=num_therm-1, step=1, value=0)

# Display the slider and the corresponding thermalisation sweep
def update_visualisation(i):
    display(ising.visualisations[i])

widgets.interactive(update_visualisation, i=slider)

    
