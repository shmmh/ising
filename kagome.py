import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from matplotlib import colors


class KagomeIsing:
    ''' A class that implements the Ising model for a Kagome Lattice
    L: size of square lattice
    J: interaction energy
    B: strength of external magnetic field
    beta: 1/kT Units of 1 and defaults to 1
    
    '''
    def __init__(self, L,J1, J2,B, beta) -> None:
        self.L = L  # Number of unit cells
        self.J1 = J1
        self.J2 = J2
        self.B = B
        self.beta = beta #/ (self.k*self.T)
        self.N = 3 * L**2
        # self.lattice = 2*np.random.randint(2, size=(self.N)) - 1 #spins either -1 or 1
        self.lattice = range(1, self.N+1)
        self.visualisations = []
        self.variance = 0
    
    def index(self, i,j,t):
        return ((i%self.L) + (((j-1)%self.L)*self.L) + (t*(self.L**2))% self.N)
    
    def nn(self,i,j,t):
        # nn of sublattice A  is (i,j,2), (i,j-1,2), (i,j,3), (i-1,j,3)
        # nn of sublattice B is (i,j,3), (i-1,j+1,3), (i,j,1), (i,j+1,1)
        # nn of sublattice C is (i,j,1), (i+1,j,1), (i,j,2), (i+1,j-1,2)
        if t == 1:
            return [(i,j,2), (i,j-1,2), (i,j,3), (i-1,j,3)]
        elif t == 2:
            return [(i,j,3), (i-1,j+1,3), (i,j,1), (i,j+1,1)]
        elif t == 3:
            return [(i,j,1), (i+1,j,1), (i,j,2), (i+1,j-1,2)]
    
    def interaction_energy(self):
        en = 0
        for i in range(self.L):
            for j in range(self.L):
                for t in range(1,4):
                        nearest_neighbours = self.nn(i,j,t)
                        for n in nearest_neighbours:
                            en += -1*self.J1 * self.lattice[self.index(*n)] * self.lattice[self.index(i,j,t)]
        return en
    
    def magnetic_energy(self):
        men = 0
        for i in range(self.L):
            for j in range(self.L):
                for t in range(1,4):
                    nearest_neighbours = self.nn(i,j,t)
                    for n in nearest_neighbours:
                        men += -1*self.B * self.lattice[self.index(i,j,t)]
        return men
    
    def total_energy(self):
        return self.interaction_energy() + self.magnetic_energy()
    
    def deltaE(self,i,j,t):
        nearest_neighbours = self.nn(i,j,t)
        f = np.sum([self.lattice[self.index(*n)] for n in nearest_neighbours])
        return -2 * (self.J1 * f + self.B) * self.lattice[self.index(i,j,t)]
    
    def flip_spin(self, i, j,t):
        self.lattice[self.index(i,j,t)] *= -1
        
    def should_flip(self, i,j,t) -> bool:
        dE = self.deltaE(i,j,t)
        rand_num = np.random.rand()
        prob = np.exp(-dE/self.beta)
        return prob > 1 or prob > rand_num
    
    def update_magnetic_field(self, b):
        self.B = b
    
    def do_thermalisation_sweep(self, n = 1):
        for i in range(self.L):
            for j in range(self.L):
                for t in range(1,4):
                    self.flip_spin(i,j,t)
                    if not self.should_flip(i,j,t):
                        # flip back if false
                        self.flip_spin(i,j,t)
                    
    
    def plot_sub_lattice(self, i,j,t,ax):
        transformations = [[2*i +j, np.sqrt(3)*j], [(2*i + j), (np.sqrt(3)*j) ], [2*i + j, np.sqrt(3)*j]][t-1]
        
        triangle_coord = np.array([[0,0], [0.5,np.sqrt(3)/2], [1,0]])[t-1]
        
        for i in range(self.L):
            for j in range(self.L):
                triangle_spins = [self.lattice[self.index(i,j,1)], self.lattice[self.index(i,j,2)], self.lattice[self.index(i,j,3)]]
                transformation_coords = lambda i,j : np.array([[2*i +j, np.sqrt(3)*j], [(2*i + j), (np.sqrt(3)*j) ], [2*i + j, np.sqrt(3)*j]])
                transformed_coords = transformation_coords(i,j) + triangle_coord
                colors = ['red' if spin == -1 else 'blue' for spin in triangle_spins]
                ax.scatter(transformed_coords[:, 0], transformed_coords[:, 1],c=colors, s=50)
                ax.annotate(f'({i},{j})', (transformed_coords[0, 0], transformed_coords[0, 1]))
            
        
    def plot_unit_cell(self, i,j,ax):
        triangle_coord = np.array([[0,0], [0.5,np.sqrt(3)/2], [1,0]])
        triangle_spins = [self.lattice[self.index(i,j,1)], self.lattice[self.index(i,j,2)], self.lattice[self.index(i,j,3)]]
        transformation_coords = lambda i,j : np.array([[2*i +j, np.sqrt(3)*j], [(2*i + j), (np.sqrt(3)*j) ], [2*i + j, np.sqrt(3)*j]])
        transformed_coords = transformation_coords(i,j) + triangle_coord
        # colors = ['red' if spin == -1 else 'blue' for spin in triangle_spins]
        colors = ['red', 'blue', 'green']
        ax.scatter(transformed_coords[:, 0], transformed_coords[:, 1],c=colors, s=100)
        
        ## annotate the spin on each point
        colors = ['red', 'blue', 'green']
        for t, coord in enumerate(transformed_coords):
            ax.annotate(f'{triangle_spins[t]}', (coord[0] + 0.1, coord[1] +0.1), color=colors[t])
        
        # for t, coord in enumerate(transformed_coords):
        #     nn = self.nn(i,j,t+1)
        #     for n in nn:
        #         i_n,j_n,_ = n
        #         trns_coord = transformation_coords(i_n,j_n) + triangle_coord
        #         for coords in trns_coord:
        #             line_pairs = ()
                 
        # ax.plot(transformed_coords[:, 0], transformed_coords[:, 1], color='black', linewidth=2)
    
    def visualise(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        for i in range(self.L+1):
            for j in range(self.L+1):
                self.plot_unit_cell(i,j,ax)
        
        # plot periodic boundary conditions
        
                
        plt.show()
                
    def get_mag_per_spin(self):
        return np.sum(self.lattice) / self.N
    
    def variance(self):
        pass
    
    def get_energy_per_spin(self):
        pass
    
    def get_succeptibility(self):
        return self.beta * np.var(self.lattice) / self.L**2
    
    def get_specific_heat(self):
        pass
    
    

        
kagome = KagomeIsing(3,1,1,0)
kagome.visualise()

    
