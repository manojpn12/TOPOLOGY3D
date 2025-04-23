import numpy as np
import matplotlib.pyplot as plt
from examples import getExampleBC
from Mesher import RectangularGridMesher, UnstructuredMesher
from projections import computeFourierMap
from material import Material
from TOuNN2 import TOuNN
from plotUtil import plotConvergence

import time
import configparser

import os
start_total = time.perf_counter()
start = start_total

mstr_files = ['./all_microstructures_3D_improved.npy']
found_file = False

for mstr_file in mstr_files:
    if os.path.exists(mstr_file):
        print(f"Found microstructure data: {mstr_file}")
        found_file = True
        break

if not found_file:
    print("No microstructure data found. Attempting to generate...")
    try:
        from regularMicrostructures3D import generateMstrImages3D
        generateMstrImages3D(step=0.05)
        print("Successfully generated microstructure data.")
    except Exception as e:
        print(f"Warning: Could not generate microstructure data: {e}")
        print("Will try again during visualization.")

# Get arguments
#configFile = './config.txt'        # original configuration file
configFile = './config2.txt'        # modified configuration file
config = configparser.ConfigParser()
config.read(configFile)

## Mesh and Boundary Conditions
meshConfig = config['MESH']
ndim = meshConfig.getint('ndim')    # number of dimensions - default is 3
nelx = meshConfig.getint('nelx')    # number of FE elements along X
nely = meshConfig.getint('nely')    # number of FE elements along Y
nelz = meshConfig.getint('nelz')    # number of FE elements along Z
elemSize = np.array(meshConfig['elemSize'].split(',')).astype(float)
start_bc = time.perf_counter()
exampleName, bcSettings, symMap = getExampleBC(1, nelx, nely, nelz, elemSize)
end_bc = time.perf_counter()
print(f"Time taken for boundary conditions: {end_bc - start_bc:.4f} seconds")

## Define grid meshe
# Examples 1-7
start_mesh = time.perf_counter()
mesh = RectangularGridMesher(ndim, nelx, nely, nelz, elemSize, bcSettings)
end_mesh = time.perf_counter()
print(f"Time taken for meshing: {end_mesh - start_mesh:.4f} seconds")

# Examples 8,9
#mesh = UnstructuredMesher(bcSettings)

## Material
materialConfig = config['MATERIAL']
E, nu = materialConfig.getfloat('E'), materialConfig.getfloat('nu')
matProp = {'physics': 'structural', 'Emax': E, 'nu': nu, 'Emin': 1e-3 * E}
start_material = time.perf_counter()
material = Material(matProp)
end_material = time.perf_counter()
print(f"Time taken for material initialization: {end_material - start_material:.4f} seconds")

## Neural Network (NN)
tounnConfig = config['TOUNN']

# Define NN configurations
nnSettings = {
    'numLayers': tounnConfig.getint('numLayers'),
    'numNeuronsPerLayer': tounnConfig.getint('hiddenDim'),
    'outputDim': tounnConfig.getint('outputDim')
}

# Adjust Fourier mapping configuration for compatibility
if tounnConfig.getboolean('fourier_isOn'):
    fourierMap = {
        'isOn': True,
        'minRadius': tounnConfig.getfloat('fourier_minRadius'),
        'maxRadius': tounnConfig.getfloat('fourier_maxRadius'),
        'numTerms': min(tounnConfig.getint('fourier_numTerms'), 50)  # Limit terms for stability
    }
    
    # Calculate input dimension based on Fourier mapping
    # Each coordinate dimension gets mapped to 2*numTerms features (sin and cos)
    nnSettings['inputDim'] = 2 * fourierMap['numTerms']
    
    # Compute the map
    start_fourier = time.perf_counter()
    fourierMap['map'] = computeFourierMap(mesh, fourierMap)
    end_fourier = time.perf_counter()
    print(f"Time taken for Fourier mapping: {end_fourier - start_fourier:.4f} seconds")
    print(f"Using Fourier mapping with {fourierMap['numTerms']} terms, input dim: {nnSettings['inputDim']}")
else:
    fourierMap = {'isOn': False}
    nnSettings['inputDim'] = mesh.ndim  # Just use the spatial dimensions directly
    print(f"No Fourier mapping, input dim: {nnSettings['inputDim']}")

# Optimization params
lossConfig = config['LOSS']
# lossMethod = {'type':'logBarrier', 't0':lossConfig.getfloat('t0'),\
#               'mu':lossConfig.getfloat('mu')}
lossMethod = {'type': 'penalty', 'alpha0': lossConfig.getfloat('alpha0'),
              'delAlpha': lossConfig.getfloat('delAlpha')}

optConfig = config['OPTIMIZATION']
optParams = {
    'maxEpochs': optConfig.getint('numEpochs'),
    'lossMethod': lossMethod,
    'learningRate': optConfig.getfloat('lr'),  # Reduced learning rate 
    'desiredVolumeFraction': optConfig.getfloat('desiredVolumeFraction'),
    'gradclip': {
        'isOn': True,  # Always use gradient clipping
        'thresh': 0.01
    },
    'penalty_weight': 1000.0  # Initial penalty for volume constraint
}

## Other projection settings
rotationalSymmetry = {'isOn': False, 'sectorAngleDeg': 90,
                      'axis': 'Z',
      'centerCoordn': np.array([20, 10, 5])}
extrusion = {'X': {'isOn': False, 'delta': 1.},
             'Y': {'isOn': False, 'delta': 1.},
             'Z': {'isOn': False, 'delta': 1.}}          

## Run optimization
plt.close('all')
savedNet = {'isAvailable': False,
            'file': './netWeights3D.pkl',
            'isDump': False}



# Instantiate the NN
start_tounn_init = time.perf_counter()
tounn = TOuNN(exampleName, mesh, material, nnSettings, symMap, fourierMap, rotationalSymmetry, extrusion)
end_tounn_init = time.perf_counter()
print(f"Time taken for TOuNN initialization: {end_tounn_init - start_tounn_init:.4f} seconds")

# Optimize the design
convgHistory = tounn.optimizeDesign(optParams,
                                    savedNet)

# Print calculation time
print(f'Total optimization time: {time.perf_counter() - start:.2F} seconds')

# Plot the convergence history
start_plot_convergence = time.perf_counter()
plotConvergence(convgHistory)
end_plot_convergence = time.perf_counter()
print(f"Time taken for convergence plot: {end_plot_convergence - start_plot_convergence:.4f} seconds")


# Plot the results with the microstructures
start_plot_topology = time.perf_counter()
tounn.plotCompositeTopology(1)
end_plot_topology = time.perf_counter()
print(f"Time taken for topology plot: {end_plot_topology - start_plot_topology:.4f} seconds")

# Show everything to the screen
plt.show(block=True)
