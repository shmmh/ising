import numpy as np
import matplotlib.pyplot as plt


class Isling:
    def __init__(self, L,J,B) -> None:
        self.L = L
        self.J = J
        self.B = B
        self.T = 273.15 # K
        self.k = 1.381e-23  # m2 kg s-2 K-1
        self.beta = 1 #/ (self.k*self.T)
        self.lattice = 2*np.random.randint(2, size=(L,L)) - 1 #spins either -1 or 1
        
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
        prob = np.exp(-dE*self.beta)
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

    def visualise(self):
        plt.figure()
        plt.imshow(self.lattice, cmap='gray', vmin=-1, vmax=1)
        plt.show()



L, B, J = 32, -0.05, 0.5
isling = Isling(L,J,B)

num_therm = 30

isling.visualise()

mag_per_spin_arr = np.zeros(num_therm)

for i in range(num_therm):
    mag_per_spin = isling.do_thermalisation_sweep()
    mag_per_spin_arr[i] = mag_per_spin


# plot magnetisation per spin array against numper of thermalisation sweeps
plt.plot(mag_per_spin_arr)


print(mag_per_spin_arr[-1])


isling.visualise()

    
