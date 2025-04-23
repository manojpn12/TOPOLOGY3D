import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import time
import os

class SingleParamMicrostructures3D:
    def __init__(self, nelx, nely, nelz, eps=0.001):
        self.nelx, self.nely, self.nelz, self.eps = nelx, nely, nelz, eps

    def cube(self, t):
        """
        3D hollow cube with wall thickness t as shown in the image
        t: Thickness parameter between 0 and 1
        """
        microStr = np.ones((self.nelx, self.nely, self.nelz)) * self.eps
        t_scaled = max(1, int(t * min(self.nelx, self.nely, self.nelz) / 3))
        
        # Create the outer shell (box)
        for x in range(self.nelx):
            for y in range(self.nely):
                for z in range(self.nelz):
                    # Check if the element is part of the outer shell with thickness t_scaled
                    if (x < t_scaled or x >= self.nelx - t_scaled or 
                        y < t_scaled or y >= self.nely - t_scaled or 
                        z < t_scaled or z >= self.nelz - t_scaled):
                        microStr[x, y, z] = 1.0
        
        return microStr

    def Xbox(self, t):
        """
        3D X-pattern inside a box as shown in the image
        t: Thickness parameter between 0 and 1
        """
        microStr = np.ones((self.nelx, self.nely, self.nelz)) * self.eps
        t_scaled = max(1, int(t * min(self.nelx, self.nely, self.nelz) / 4))
        
        # First create the outer shell (box)
        for x in range(self.nelx):
            for y in range(self.nely):
                for z in range(self.nelz):
                    # Check if the element is part of the outer shell
                    if (x < t_scaled or x >= self.nelx - t_scaled or 
                        y < t_scaled or y >= self.nely - t_scaled or 
                        z < t_scaled or z >= self.nelz - t_scaled):
                        microStr[x, y, z] = 1.0
        
        # Add X pattern in the 2D projection (as shown in the image)
        # Calculate center of the domain
        cx, cy = self.nelx / 2, self.nely / 2
        beam_thickness = max(1, int(t * min(self.nelx, self.nely) / 3))
        
        for x in range(self.nelx):
            for y in range(self.nely):
                # Calculate distance from this point to the diagonals
                # Diagonal 1: y = x (from top-left to bottom-right)
                d1 = abs(y - x)
                # Diagonal 2: y = nelx - x (from top-right to bottom-left)
                d2 = abs(y - (self.nelx - 1 - x))
                
                # If this point is close enough to either diagonal, it's part of the X
                if d1 <= beam_thickness or d2 <= beam_thickness:
                    # Apply this X pattern to all z-slices
                    for z in range(self.nelz):
                        microStr[x, y, z] = 1.0
        
        return microStr
    
    def XPlusBox(self, t):
        """
        X pattern with additional cross beams inside a box as shown in the image
        t: Thickness parameter between 0 and 1
        """
        # Start with the Xbox pattern
        microStr = self.Xbox(t)
        t_scaled = max(1, int(t * min(self.nelx, self.nely, self.nelz) / 5))
        
        # Calculate center of the domain
        cx, cy = int(self.nelx / 2), int(self.nely / 2)
        
        # Add the horizontal and vertical beams to create the + pattern
        beam_thickness = max(1, int(t * min(self.nelx, self.nely) / 4))
        
        for x in range(self.nelx):
            for y in range(self.nely):
                # Horizontal beam: fixed y = cy
                if abs(y - cy) <= beam_thickness:
                    # Apply to all z-slices
                    for z in range(self.nelz):
                        microStr[x, y, z] = 1.0
                
                # Vertical beam: fixed x = cx
                if abs(x - cx) <= beam_thickness:
                    # Apply to all z-slices
                    for z in range(self.nelz):
                        microStr[x, y, z] = 1.0
        
        return microStr

    def plotMicrostructure(self, mc):
        """
        Plot 2D slices of the 3D microstructure to match the image
        """
        mc = mc.reshape(self.nelx, self.nely, self.nelz)
        mid_z = self.nelz // 2  # Middle z-slice for visualization
        
        plt.figure(figsize=(10, 8))
        plt.imshow(mc[:, :, mid_z].T, cmap='gray', 
                   interpolation='none', origin='lower')
        plt.title(f'Microstructure - Volume Fraction: {np.mean(mc):.4f}')
        plt.colorbar(label='Density')
        plt.tight_layout()
        plt.show()
        
        # Also show 3D visualization
        self.plot3DMicrostructure(mc)
        
    def plot3DMicrostructure(self, mc):
        """
        Create a 3D visualization of the microstructure
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get points where material exists (density > 0.5)
        points = np.where(mc > 0.5)
        x, y, z = points[0], points[1], points[2]
        
        # Limit points for visualization (maximum 5000 points)
        if len(x) > 5000:
            idx = np.random.choice(len(x), 5000, replace=False)
            x, y, z = x[idx], y[idx], z[idx]
        
        # Plot the points
        ax.scatter(x, y, z, c='blue', marker='o', alpha=0.7, s=10)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Microstructure Visualization')
        
        # Set axis limits
        ax.set_xlim(0, self.nelx)
        ax.set_ylim(0, self.nely)
        ax.set_zlim(0, self.nelz)
        
        plt.tight_layout()
        plt.show()

def generateMstrImages3D(step=0.05):
    """
    Generate 3D microstructure images with varying thickness parameters
    and save them to a numpy file.
    
    step: Thickness increment (default: 0.05) between 0 and 1
    """
    t = np.arange(0.05, 1., step)  # Thickness steps
    nelx, nely, nelz = 20, 20, 20  # Grid size for 3D microstructures
    M = SingleParamMicrostructures3D(nelx, nely, nelz)  # Initialize 3D microstructures class
    
    # Dictionary of 3D microstructures
    mstrFn = {
        '0': lambda t: M.cube(t),
        '1': lambda t: M.Xbox(t),
        '2': lambda t: M.XPlusBox(t),
    }

    # Initialize the storage array for all microstructures
    microstructures = np.zeros((len(mstrFn), t.shape[0], nelx, nely, nelz))

    # Generate microstructure images
    for m, mfn in mstrFn.items():  # Iterate over microstructure types
        for i, thickness in enumerate(t):  # Iterate over thickness values
            print(f"Generating microstructure {m}, step {i}")
            microstructures[int(m), i] = mfn(thickness)  # Generate and store

    # Save the generated 3D microstructures to a file
    np.save('all_microstructures_3D_updated.npy', microstructures)
    print("3D microstructures saved in a single file.")

def test3D():
    """
    Test the 3D microstructures and visualize them
    """
    plt.close('all')
    nelx, nely, nelz = 20, 20, 20  # Smaller size for faster testing
    M = SingleParamMicrostructures3D(nelx, nely, nelz)
    
    # Test each microstructure type
    for mstr_type, mstr_name, thickness in zip(range(3), ['Cube', 'Xbox', 'XPlusBox'], [0.2, 0.3, 0.4]):
        t = thickness  # Different thickness for different microstructures
        
        if mstr_type == 0:
            mstr = M.cube(t)
        elif mstr_type == 1:
            mstr = M.Xbox(t)
        else:
            mstr = M.XPlusBox(t)
            
        print(f"\nTesting microstructure: {mstr_name}")
        M.plotMicrostructure(mstr)
        
        # Get volume fraction
        vf = np.mean(mstr)
        print(f'Volume Fraction (vf): {vf:.4f}')

# Run tests
if __name__ == "__main__":
    # Test the microstructures
    test3D()
    
    # Generate and save microstructure library
    generateMstrImages3D(step=0.1)  # Use larger step for faster generation