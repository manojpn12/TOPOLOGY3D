import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from jax.numpy import index_exp as index
import time


class RectangularGridMesher:

    def __init__(self, ndim, nelx, nely, nelz, elemSize, bcSettings):
        start_mesh_init = time.perf_counter()
        self.meshType = 'gridMesh'
        self.ndim = ndim
        self.nelx = nelx
        self.nely = nely
        self.nelz = nelz
        self.elemSize = elemSize
        self.bcSettings = bcSettings
        self.numElems = self.nelx * self.nely * self.nelz
        self.elemVolume = self.elemSize[0] * self.elemSize[1] * self.elemSize[2] * jnp.ones(self.numElems)  # all same areas for grid
        self.totalMeshVolume = jnp.sum(self.elemVolume)

        # Number of nodes
        self.numNodes = (self.nelx + 1) * (self.nely + 1) * (self.nelz + 1)
        #print(self.numNodes)
        #print(self.nelx)
        #print(self.nely)

        # Grid is quad mesh
        self.nodesPerElem = 8

        # Number of degree of freedom (x,y)
        self.ndof = self.bcSettings['dofsPerNode'] * self.numNodes

        # Create mesh structure
        self.edofMat, self.nodeIdx, self.elemNodes, self.nodeXYZ, self.bb = self.getMeshStructure()
        end_mesh_structure = time.perf_counter()
        print(f"Time taken for mesh initialization: {end_mesh_structure - start_mesh_init:.4f} seconds")

        # Get x,y centers of the elements
        self.elemCenters = self.nodeXYZ[self.elemNodes].mean(axis=1)

        # Plot the mesh and the boundary conditions
        # Plot the mesh grid of the elements
        #print(np.shape(self.nodeXY[:, 0]))-1/2

        ## Plotting the 3D mesh
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ## Plot the nodes in 3D
        ax.scatter(self.nodeXYZ[:, 0], self.nodeXYZ[:, 1], self.nodeXYZ[:, 2], color='black', label="Nodes")

        ## Plot element centers in 3D
        ax.scatter(self.elemCenters[:, 0], self.elemCenters[:, 1], self.elemCenters[:, 2], color='green', label="Element Centers")

        # Set grid limits based on mesh dimensions
        ax.set_xlim([0, self.nelx * 1])
        ax.set_ylim([0, self.nely * 1])
        ax.set_zlim([0, self.nelz * 1])  # Z limit added for 3D plotting

        # Add gridlines
        ax.grid(color='k', linestyle='-', linewidth=1)

        print("Force on nodes:", self.bcSettings['forceNodes'], flush=True)

        # Plot the fixed nodes in 3D
        ax.scatter(self.nodeXYZ[self.bcSettings['fixedNodes'], 0], 
                self.nodeXYZ[self.bcSettings['fixedNodes'], 1], 
                self.nodeXYZ[self.bcSettings['fixedNodes'], 2], color='blue', label="Fixed Nodes")

        # Plot the force nodes
        #f_node = int((self.bcSettings['forceNodes'])/2)-1

        ## Collect forces in 3D
        x_nodes, y_nodes, z_nodes = [], [], []

        # First, convert the forceNodes to actual node indices if they're DOFs
        if hasattr(self.bcSettings['forceNodes'], '__len__'):  # Check if forceNodes is a list
            force_nodes = list(self.bcSettings['forceNodes'])
        else:
            # Single force node
            force_nodes = [self.bcSettings['forceNodes']]

        # Plot the force nodes (simplified)
        ax.scatter(self.nodeXYZ[force_nodes, 0], 
                self.nodeXYZ[force_nodes, 1], 
                self.nodeXYZ[force_nodes, 2], 
                color='red', marker='^', s=100, label="Force Nodes")

        # Show legend
        ax.legend()

        # Show the plot
        plt.show()


        # Set the boundary conditions
        self.processBoundaryCondition()
        end_bc_process = time.perf_counter()
        print(f"Boundary condition processing time: {end_bc_process - end_mesh_structure:.4f} seconds")


        # ?
        self.BMatrix = self.getBMatrix(0., 0., 0.)

        # Plot settings
        self.fig, self.ax = plt.subplots()
        self.bb = {'xmax': self.nelx * self.elemSize[0], 'xmin': 0.,
                   'ymax': self.nely * self.elemSize[1], 'ymin': 0.,
                   'zmax': self.nelz * self.elemSize[2], 'zmin': 0.}
        
        # End timing for mesh initialization
        end_mesh_init = time.perf_counter()
        print(f"Total mesh initialization time: {end_mesh_init - start_mesh_init:.4f} seconds")


    def getBMatrix(self, xi, eta, zeta):
        #
        dx, dy, dz = self.elemSize[0], self.elemSize[1], self.elemSize[2]  # Element dimensions

        # Shape function derivatives in ξ, η, ζ directions
        r1 = np.array([(2. * (eta / 4. - 1. / 4.)) / dx, -(2. * (eta / 4. - 1. / 4)) / dx,
                    (2. * (eta / 4. + 1. / 4)) / dx, -(2. * (eta / 4. + 1. / 4)) / dx,
                    (2. * (eta / 4. - 1. / 4.)) / dx, -(2. * (eta / 4. - 1. / 4)) / dx,
                    (2. * (eta / 4. + 1. / 4)) / dx, -(2. * (eta / 4. + 1. / 4)) / dx]).reshape(-1)

        r2 = np.array([(2. * (xi / 4. - 1. / 4.)) / dy, -(2. * (xi / 4. + 1. / 4)) / dy,
                    (2. * (xi / 4. + 1. / 4)) / dy, -(2. * (xi / 4. - 1. / 4)) / dy,
                    (2. * (xi / 4. - 1. / 4.)) / dy, -(2. * (xi / 4. + 1. / 4)) / dy,
                    (2. * (xi / 4. + 1. / 4)) / dy, -(2. * (xi / 4. - 1. / 4)) / dy]).reshape(-1)

        r3 = np.array([(2. * (zeta / 4. - 1. / 4.)) / dz, -(2. * (zeta / 4. + 1. / 4)) / dz,
                    (2. * (zeta / 4. + 1. / 4)) / dz, -(2. * (zeta / 4. - 1. / 4)) / dz,
                    (2. * (zeta / 4. - 1. / 4.)) / dz, -(2. * (zeta / 4. + 1. / 4)) / dz,
                    (2. * (zeta / 4. + 1. / 4)) / dz, -(2. * (zeta / 4. - 1. / 4)) / dz]).reshape(-1)

        # 3D B-Matrix (6x24) for hexahedral elements
        B = [[r1[0], 0., 0., r1[1], 0., 0., r1[2], 0., 0., r1[3], 0., 0., r1[4], 0., 0., r1[5], 0., 0., r1[6], 0., 0., r1[7], 0., 0.],
            [0., r2[0], 0., 0., r2[1], 0., 0., r2[2], 0., 0., r2[3], 0., 0., r2[4], 0., 0., r2[5], 0., 0., r2[6], 0., 0., r2[7], 0.],
            [0., 0., r3[0], 0., 0., r3[1], 0., 0., r3[2], 0., 0., r3[3], 0., 0., r3[4], 0., 0., r3[5], 0., 0., r3[6], 0., 0., r3[7]],
            [r2[0], r1[0], 0., r2[1], r1[1], 0., r2[2], r1[2], 0., r2[3], r1[3], 0., r2[4], r1[4], 0., r2[5], r1[5], 0., r2[6], r1[6], 0., r2[7], r1[7], 0.],
            [0., r3[0], r2[0], 0., r3[1], r2[1], 0., r3[2], r2[2], 0., r3[3], r2[3], 0., r3[4], r2[4], 0., r3[5], r2[5], 0., r3[6], r2[6], 0., r3[7], r2[7]],
            [r3[0], 0., r1[0], r3[1], 0., r1[1], r3[2], 0., r1[2], r3[3], 0., r1[3], r3[4], 0., r1[4], r3[5], 0., r1[5], r3[6], 0., r1[6], r3[7], 0., r1[7]]]

        return jnp.array(B)


    def getMeshStructure(self):
        start_structure = time.perf_counter()
        # edofMat: Connectivity matrix for each element
        # Returns edofMat: array of size (numElemsX8) with the global dof of each element
        # idx: A tuple informing the position for assembly of computed entries

        # n is the number of degrees of freedom per element
        n = self.bcSettings['dofsPerNode'] * self.nodesPerElem  # 3 dof * 8 nodes per element = 24

        # Connectivity matrix
        # Each element is one line ([4 nodes * 2] dof collumns)
        edofMat = np.zeros((self.nelx * self.nely * self.nelz, n), dtype=int)

        # As in structural (2 degrees of freedom)
        if self.bcSettings['dofsPerNode'] == 3:
            # Iteraring over the elements in x
            for elx in range(self.nelx):
                # Iterating over the elements in y
                for ely in range(self.nely):
                    # Iterating over the elements in z
                    for elz in range(self.nelz):
                    # Get element
                        el = elz + ely * self.nelz + elx * self.nely * self.nelz  # Element index
                    #print("el: ", el, flush=True)

                        # Compute global node indices
                        n1 = (self.nely + 1) * (self.nelz + 1) * elx + (self.nelz + 1) * ely + elz
                        n2 = (self.nely + 1) * (self.nelz + 1) * (elx + 1) + (self.nelz + 1) * ely + elz
                        n3 = (self.nely + 1) * (self.nelz + 1) * (elx + 1) + (self.nelz + 1) * (ely + 1) + elz
                        n4 = (self.nely + 1) * (self.nelz + 1) * elx + (self.nelz + 1) * (ely + 1) + elz
                        n5 = n1 + 1
                        n6 = n2 + 1
                        n7 = n3 + 1
                        n8 = n4 + 1

                        # Assign DOFs for each node (3 DOFs per node in 3D)
                        edofMat[el, :] = np.array([
                            3 * n1, 3 * n1 + 1, 3 * n1 + 2,  # Node 1
                            3 * n2, 3 * n2 + 1, 3 * n2 + 2,  # Node 2
                            3 * n3, 3 * n3 + 1, 3 * n3 + 2,  # Node 3
                            3 * n4, 3 * n4 + 1, 3 * n4 + 2,  # Node 4
                            3 * n5, 3 * n5 + 1, 3 * n5 + 2,  # Node 5
                            3 * n6, 3 * n6 + 1, 3 * n6 + 2,  # Node 6
                            3 * n7, 3 * n7 + 1, 3 * n7 + 2,  # Node 7
                            3 * n8, 3 * n8 + 1, 3 * n8 + 2   # Node 8
                        ])

                    #print("edofmat: ", edofMat[el, :], flush=True)
                    #input("e")

        # As in thermal (1 degree of freedom)
        elif self.bcSettings['dofsPerNode'] == 1:  # Thermal analysis (1 DOF per node)
            nodenrs = np.reshape(np.arange(0, self.ndof), (1 + self.nelx, 1 + self.nely, 1 + self.nelz))
            edofVec = np.reshape(nodenrs[0:-1, 0:-1, 0:-1] + 1, self.numElems, 'F')
            edofMat = np.matlib.repmat(edofVec, 8, 1).T + \
                        np.matlib.repmat(np.array([0, 1, self.nely + 1, self.nely + 2,
                                                    (self.nely + 1) * (self.nelz + 1),
                                                    (self.nely + 1) * (self.nelz + 1) + 1,
                                                    (self.nely + 1) * (self.nelz + 1) + self.nely + 1,
                                                    (self.nely + 1) * (self.nelz + 1) + self.nely + 2]),
                                        self.numElems, 1)

        # Define node index for FEM matrix assembly
        iK = tuple(np.kron(edofMat, np.ones((n, 1))).flatten().astype(int))
        jK = tuple(np.kron(edofMat, np.ones((1, n))).flatten().astype(int))
        nodeIdx = (iK, jK)
            #print(nodeIdx)
            #print(iK)
            #print(jK)
            #exit()

        # Define nodes per element
        elemNodes = np.zeros((self.numElems, self.nodesPerElem), dtype=int)
    
        # Loop through elements in 3D
        for elx in range(self.nelx):
            for ely in range(self.nely):
                for elz in range(self.nelz):
                    el = elz + ely * self.nelz + elx * self.nely * self.nelz  # Element index
                    
                    # Compute global node indices
                    n1 = (self.nely + 1) * (self.nelz + 1) * elx + (self.nelz + 1) * ely + elz
                    n2 = (self.nely + 1) * (self.nelz + 1) * (elx + 1) + (self.nelz + 1) * ely + elz
                    n3 = (self.nely + 1) * (self.nelz + 1) * (elx + 1) + (self.nelz + 1) * (ely + 1) + elz
                    n4 = (self.nely + 1) * (self.nelz + 1) * elx + (self.nelz + 1) * (ely + 1) + elz
                    n5 = n1 + 1
                    n6 = n2 + 1
                    n7 = n3 + 1
                    n8 = n4 + 1

                    # Assign 8-node connectivity
                    elemNodes[el, :] = np.array([n1, n2, n3, n4, n5, n6, n7, n8])

        # Bounding box definition for 3D
        bb = {
            'xmin': 0., 'xmax': self.nelx * self.elemSize[0],
            'ymin': 0., 'ymax': self.nely * self.elemSize[1],
            'zmin': 0., 'zmax': self.nelz * self.elemSize[2]  # Added Z bounds
        }

        # Define [x, y, z] positions of the nodes
        nodeXYZ = np.zeros((self.numNodes, 3))  # Now 3D
        ctr = 0
        for i in range(self.nelx + 1):
            for j in range(self.nely + 1):
                for k in range(self.nelz + 1):  # Added Z loop
                    nodeXYZ[ctr, 0] = self.elemSize[0] * i  # X-coordinate
                    nodeXYZ[ctr, 1] = self.elemSize[1] * j  # Y-coordinate
                    nodeXYZ[ctr, 2] = self.elemSize[2] * k  # Z-coordinate
                    ctr += 1

        end_structure = time.perf_counter()
        print(f"Time taken for mesh structure: {end_structure - start_structure:.4f} seconds")

        # Return connectivity, node index, element nodes, node positions, bounding box
        return edofMat, nodeIdx, elemNodes, nodeXYZ, bb

    def generatePoints(self, res=1):
        # args: Mesh is dictionary containing nelx, nely, elemSize...
        # res is the number of points per elem
        # returns an array of size (numpts X 2)
        xyz = np.zeros((res ** 3 * self.numElems, 3))  # Create an empty array for (X, Y, Z) points
        ctr = 0
        
        for i in range(res * self.nelx):  # Loop over X elements
            for j in range(res * self.nely):  # Loop over Y elements
                for k in range(res * self.nelz):  # Loop over Z elements
                    xyz[ctr, 0] = self.elemSize[0] * (i + 0.5) / res  # X position
                    xyz[ctr, 1] = self.elemSize[1] * (j + 0.5) / res  # Y position
                    xyz[ctr, 2] = self.elemSize[2] * (k + 0.5) / res  # Z position
                    ctr += 1
        
        return xyz

    def processBoundaryCondition(self):
        start_bc = time.perf_counter()
        # Initialize force vector with zeros
        force = np.zeros((self.ndof,))
        
        # Process the force nodes
        force_nodes = self.bcSettings['forceNodes']
        
        if hasattr(force_nodes, '__len__'):
            # For multiple force nodes
            for node in force_nodes:
                # For 3D, apply force in the y-direction (DOF index 1)
                force_dof = self.bcSettings['dofsPerNode'] * node + 1
                force[force_dof] = self.bcSettings['forceMagnitude']
        else:
            # Single force node
            force_dof = self.bcSettings['dofsPerNode'] * force_nodes + 1  # Y-direction by default
            force[force_dof] = self.bcSettings['forceMagnitude']
        
        # Fixed DOFs - convert node indices to DOFs
        fixed_nodes = self.bcSettings['fixedNodes']
        fixed_dofs = []
        for node in fixed_nodes:
            for dof_idx in range(self.bcSettings['dofsPerNode']):
                fixed_dofs.append(self.bcSettings['dofsPerNode'] * node + dof_idx)
        fixed = np.array(fixed_dofs)
        
        # Compute free DOFs
        free = np.setdiff1d(np.arange(self.ndof), fixed)
        
        # Store boundary conditions
        self.bc = {'force': force.reshape(-1, 1), 'fixed': fixed, 'free': free}

        end_bc = time.perf_counter()
        print(f"Time taken for boundary conditions: {end_bc - start_bc:.4f} seconds")
            
        # Additional check - is the stiffness matrix properly constrained?
        num_fixed_dofs = len(fixed)
        min_required = 6  # For 3D, need at least 6 constraints to prevent rigid body motion
        if num_fixed_dofs < min_required:
            print(f"WARNING: Only {num_fixed_dofs} DOFs are fixed. At least {min_required} are needed to prevent rigid body motion.")

    def plotFieldOnMesh(self, field, titleStr):
        """Plot 3D field values on the mesh with guaranteed consistent size"""
        try:
            # Close any existing plots to start fresh (this is key to avoiding the shrinking)
            if hasattr(self, 'plot_fig') and plt.fignum_exists(self.plot_fig.number):
                plt.close(self.plot_fig)
            
            # Create a completely new figure each time with fixed size
            self.plot_fig = plt.figure(figsize=(10, 8), dpi=100)
            self.plot_ax = self.plot_fig.add_subplot(111, projection='3d')
            
            # Set window title
            self.plot_fig.canvas.manager.set_window_title("Optimization Progress")
            
            # Extract X, Y, Z coordinates
            X = self.elemCenters[:, 0]
            Y = self.elemCenters[:, 1]
            Z = self.elemCenters[:, 2]
            
            # Store axis limits if not already stored
            if not hasattr(self, 'axis_limits'):
                x_margin = (np.max(X) - np.min(X)) * 0.1
                y_margin = (np.max(Y) - np.min(Y)) * 0.1
                z_margin = (np.max(Z) - np.min(Z)) * 0.1
                
                self.axis_limits = {
                    'x_min': np.min(X) - x_margin,
                    'x_max': np.max(X) + x_margin,
                    'y_min': np.min(Y) - y_margin,
                    'y_max': np.max(Y) + y_margin,
                    'z_min': np.min(Z) - z_margin,
                    'z_max': np.max(Z) + z_margin
                }
            
            # Scatter plot of field values
            scatter = self.plot_ax.scatter(X, Y, Z, c=field, cmap='viridis', marker='o', s=80)
            
            # Set fixed axis limits
            self.plot_ax.set_xlim(self.axis_limits['x_min'], self.axis_limits['x_max'])
            self.plot_ax.set_ylim(self.axis_limits['y_min'], self.axis_limits['y_max'])
            self.plot_ax.set_zlim(self.axis_limits['z_min'], self.axis_limits['z_max'])
            
            # Turn off autoscaling explicitly
            self.plot_ax.autoscale(enable=False)
            
            # Set labels and title
            self.plot_ax.set_xlabel('X', fontsize=12)
            self.plot_ax.set_ylabel('Y', fontsize=12)
            self.plot_ax.set_zlabel('Z', fontsize=12)
            self.plot_ax.set_title(titleStr, fontsize=14)
            
            # Add colorbar
            self.colorbar = self.plot_fig.colorbar(scatter, ax=self.plot_ax, label="Density", pad=0.1)
            
            # Set up fixed aspect ratio
            self.plot_ax.set_box_aspect([1, 1, 1])
            
            # Disable tight layout which can cause resizing
            # Instead use a fixed subplot position
            self.plot_ax.set_position([0.1, 0.1, 0.8, 0.8])
            
            # Draw and show without using tight_layout
            self.plot_fig.canvas.draw()
            
            # For non-blocking display
            plt.ion()
            plt.show(block=False)
            plt.pause(0.1)  # Small pause to ensure display
            
            # Save figure every 10 epochs (optional)
            try:
                epoch_str = titleStr.split(',')[0].split(':')[1].strip()
                epoch_num = int(epoch_str.split('/')[0])
                if epoch_num % 10 == 0:
                    import os
                    if not os.path.exists('./optimization_plots'):
                        os.makedirs('./optimization_plots')
                    plt.savefig(f'./optimization_plots/epoch_{epoch_num:03d}.png', dpi=150)
            except:
                pass
            
        except Exception as e:
            print(f"Error in plotFieldOnMesh: {e}")
            import traceback
            traceback.print_exc()
        # self.fig.canvas.draw()


class UnstructuredMesher:
    def __init__(self, bcFiles):
        self.bcFiles = bcFiles
        self.meshProp = {}
        self.meshType = 'unstructuredMesh'
        self.ndim = 3  # 3D structures
        self.dofsPerNode = 3  # structural

        self.readMeshData()
        self.fig = plt.figure()  # Create a figure for plotting
        self.ax = self.fig.add_subplot(111, projection='3d')

    def readMeshData(self):
        # Only structural mesh
        self.bc = {}

        # Grid quad mesh
        self.nodesPerElem = 8

        # Force
        with open(self.bcFiles['forceFile']) as f:
            self.bc['force'] = np.array([float(line.rstrip()) for line in f]).reshape(-1)
        self.ndof = self.bc['force'].shape[0]  # Total DOFs
        self.numNodes = int(self.ndof / 3)  # Number of nodes

        # Fixed
        with open(self.bcFiles['fixedFile']) as f:
            self.bc['fixed'] = np.array([int(line.rstrip()) for line in f]).reshape(-1)
        self.bc['free'] = np.setdiff1d(np.arange(self.ndof), self.bc['fixed'])

        # Node XY
        self.nodeXYZ = np.zeros((self.numNodes, self.ndim))
        ctr = 0
        f = open(self.bcFiles['nodeXYZFile'])
        for line in f:
            self.nodeXYZ[ctr, :] = line.rstrip().split('\t')
            ctr += 1

        # edofMat
        ctr = 0
        f = open(self.bcFiles['elemNodesFile'])
        self.numElems = int(f.readline().rstrip())
        self.elemSize = np.zeros((3))
        self.elemSize[0], self.elemSize[1], self.elemSize[2] = \
           map(float, f.readline().rstrip().split('\t'))

        # volumes
        self.elemVolume = self.elemSize[0] * self.elemSize[1] * self.elemSize[2] * jnp.ones(self.numElems)
        self.totalMeshVolume = jnp.sum(self.elemVolume)

        # Nodes per element in 3D (Hexahedron has 8 nodes)
        self.nodesPerElem = 8
        self.elemNodes = np.zeros((self.numElems, self.nodesPerElem))
        self.dofsPerNode = 3  # XYZ displacements

        # Element DOF connectivity matrix (edofMat)
        self.edofMat = np.zeros((self.numElems, self.nodesPerElem * self.dofsPerNode))

        for line in f:
            self.elemNodes[ctr, :] = line.rstrip().split('\t')
            self.edofMat[ctr, :] = np.array([
                [3 * self.elemNodes[ctr, i], 
                3 * self.elemNodes[ctr, i] + 1, 
                3 * self.elemNodes[ctr, i] + 2]
                for i in range(self.nodesPerElem)
            ]).reshape(-1)
            ctr += 1

        self.edofMat = self.edofMat.astype(int)
        self.elemNodes = self.elemNodes.astype(int)

        # Compute element centers in 3D
        self.elemCenters = self.nodeXYZ[self.elemNodes].mean(axis=1)
        for elem in range(self.numElems):
            nodes = ((self.edofMat[elem, 0::3] + 3) / 3).astype(int) - 1
            for i in range(8):
                self.elemCenters[elem, 0] += 0.125 * self.nodeXYZ[nodes[i], 0]
                self.elemCenters[elem, 1] += 0.125 * self.nodeXYZ[nodes[i], 1]
                self.elemCenters[elem, 2] += 0.125 * self.nodeXYZ[nodes[i], 2]

        # Element vertices coordinates in 3D
        self.elemVertices = np.zeros((self.numElems, self.nodesPerElem, 3))
        for elem in range(self.numElems):
            nodes = ((self.edofMat[elem, 0::3] + 3) / 3).astype(int) - 1
            self.elemVertices[elem, :, 0] = self.nodeXYZ[nodes, 0]
            self.elemVertices[elem, :, 1] = self.nodeXYZ[nodes, 1]
            self.elemVertices[elem, :, 2] = self.nodeXYZ[nodes, 2]

        # Index mapping for global stiffness matrix assembly in 3D
        iK = np.kron(self.edofMat, np.ones((24, 1))).flatten().astype(int)
        jK = np.kron(self.edofMat, np.ones((1, 24))).flatten().astype(int)
        self.nodeIdx = index[iK, jK]

        # Bounding box of the mesh in 3D
        self.bb = {}
        self.bb['xmin'], self.bb['xmax'] = np.min(self.nodeXYZ[:, 0]), np.max(self.nodeXYZ[:, 0])
        self.bb['ymin'], self.bb['ymax'] = np.min(self.nodeXYZ[:, 1]), np.max(self.nodeXYZ[:, 1])
        self.bb['zmin'], self.bb['zmax'] = np.min(self.nodeXYZ[:, 2]), np.max(self.nodeXYZ[:, 2])

    def generatePoints(self, res=1, includeEndPts=False):
        if (includeEndPts):
            endPts = 2
            resMin, resMax = 0, res + 2
        else:
            endPts = 0
            resMin, resMax = 1, res + 1
        points = np.zeros((self.numElems * (res + endPts) ** 3, 3))
        ctr = 0
        for elm in range(self.numElems):
            nodes = self.elemNodes[elm, :]
            xmin, xmax = np.min(self.nodeXY[nodes, 0]), np.max(self.nodeXY[nodes, 0])
            ymin, ymax = np.min(self.nodeXY[nodes, 1]), np.max(self.nodeXY[nodes, 1])
            zmin, zmax = np.min(self.nodeXY[nodes, 2]), np.max(self.nodeXY[nodes, 2])
            delX = (xmax - xmin) / (res + 1.)
            delY = (ymax - ymin) / (res + 1.)
            delZ = (zmax - zmin) / (res + 1.)
            for rx in range(resMin, resMax):
                xv = xmin + rx * delX
                for ry in range(resMin, resMax):
                    yv = ymin + ry * delY
                    for rz in range(resMin, resMax):
                        points[ctr, 0] = xv
                        points[ctr, 1] = yv
                        points[ctr, 2] = zmin + rz * delZ
                        ctr += 1
        return points

    def plotFieldOnMesh(self, field, titleStr, res=1):

        x = self.nodeXY[:, 0]
        y = self.nodeXY[:, 1]
        z = self.nodeXY[:, 2]


        def hexaplot(x, y, z, hexahedrons, values, ax=None, **kwargs):
            if ax is None:
                ax = plt.gca(projection='3d')  # Get 3D axis
            
            xyz = np.c_[x, y, z]  # Stack coordinates
            verts = [xyz[hexa] for hexa in hexahedrons]  # Get vertices

            # Create 3D polygon collection for hex elements
            pc = Poly3DCollection(verts, **kwargs)
            pc.set_array(values)  # Assign field values to color mapping
            ax.add_collection3d(pc)

            return pc

        # Create 3D figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot hexahedral mesh with field values
        pc = hexaplot(x, y, z, np.asarray(self.elemNodes), -field, ax=ax,
                    edgecolor="crimson", cmap="gray")

        # Labels and title
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title(titleStr)

        # Colorbar
        # self.fig.canvas.draw()
