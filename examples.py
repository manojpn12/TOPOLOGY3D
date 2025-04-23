import numpy as np
import time
import matplotlib.pyplot as plt

## Examples of pre-defined shapes
def getExampleBC(example, nelx, nely, nelz, elemSize):

    ## Tip cantilever
    if example == 1:
        exampleName = 'TipCantilever3D'
        
        total_nodes = (nelx + 1) * (nely + 1) * (nelz + 1)
        print(f"Total nodes: {total_nodes}")
        
        # Fixed nodes - all nodes on left face (x=0)
        left_face_nodes = []
        i = 0  # Left face (x=0)
        for j in range(nely + 1):
            for k in range(nelz + 1):
                # Correct 3D node indexing formula
                node = k + j*(nelz+1) + i*(nely+1)*(nelz+1)
                left_face_nodes.append(node)
        
        fixed_nodes = np.array(left_face_nodes)
        print(f"Fixed nodes: {len(fixed_nodes)} nodes on the left face")
        
        # Find nodes on the right face mid-line for force application
        i = nelx  # Right face (x=nelx)
        mid_y = nely // 2
        
        # Collect all nodes along the mid-line in z-direction
        right_mid_nodes = []
        for k in range(nelz + 1):
            node = k + mid_y*(nelz+1) + i*(nely+1)*(nelz+1)
            right_mid_nodes.append(node)
        
        # Total force and per-node force
        total_force = -1000.0
        force_per_node = total_force / len(right_mid_nodes)
        
        print(f"Force will be applied to {len(right_mid_nodes)} nodes along the mid-line")
        print("Force node indices:", right_mid_nodes)
        print(f"Force per node: {force_per_node}")
        
        bcSettings = {
            'fixedNodes': fixed_nodes,
            'forceMagnitude': force_per_node,
            'forceNodes': right_mid_nodes,  # Multiple nodes along mid-line
            'dofsPerNode': 3,  # 3 DOFs per node
            'forceDirection': 1  # Y-direction (second DOF)
        }
        
        # Detailed force application logging
        print("Force Application Details:")
        for node in right_mid_nodes:
            force_dof = bcSettings['dofsPerNode'] * node + bcSettings['forceDirection']
            print(f"Node {node}, DOF {force_dof}, Force: {force_per_node}")
        
        symMap = {'XAxis': {'isOn': False, 'midPt': 0.5 * nely * elemSize[1]},
                'YAxis': {'isOn': False, 'midPt': 0.5 * nelx * elemSize[0]},
                'ZAxis': {'isOn': False, 'midPt': 0.5 * nelz * elemSize[2]}}
    
    ## Mid-cantilever
    elif example == 2:
        exampleName = 'MidCantilever'
        bcSettings = {'fixedNodes': np.arange(0, 2 * (nely + 1), 1),
                      'forceMagnitude': -1.,
                      'forceNodes': 2 * (nelx + 1) * (nely + 1) - (nely + 1),
                      'dofsPerNode': 2}
        symMap = {'XAxis': {'isOn': True, 'midPt': 0.5 * nely * elemSize[1]},
                  'YAxis': {'isOn': False, 'midPt': 0.5 * nelx * elemSize[0]}}

    ## MBB-Beam
    elif example == 3:
        exampleName = 'MBBBeam'
        fn = np.union1d(np.arange(0, 2 * (nely + 1), 2), 2 * (nelx + 1) * (nely + 1) - 2 * (nely + 1) + 1)
        bcSettings = {'fixedNodes': fn,
                      'forceMagnitude': -1.,
                      'forceNodes': 2 * (nely + 1) + 1,
                      'dofsPerNode': 2}
        symMap = {'XAxis': {'isOn': False, 'midPt': 0.5 * nely * elemSize[1]},
                  'YAxis': {'isOn': False, 'midPt': 0.5 * nelx * elemSize[0]}}

    ## Michell
    elif example == 4:
        exampleName = 'Michell'
        fn = np.array([0, 1, 2 * (nelx + 1) * (nely + 1) - 2 * nely])
        bcSettings = {'fixedNodes': fn,
                      'forceMagnitude': -1.,
                      'forceNodes': nelx * (nely + 1) + 1,
                      'dofsPerNode': 2}
        symMap = {'XAxis': {'isOn': False, 'midPt': 0.5 * nely * elemSize[1]},
                  'YAxis': {'isOn': True, 'midPt': 0.5 * nelx * elemSize[0]}}

    ## Distributed-MBB (Bridge)
    elif example == 5:
        exampleName = 'Bridge'
        fixn = np.array([0, 1, 2 * (nelx + 1) * (nely + 1) - 2 * nely + 1,
                         2 * (nelx + 1) * (nely + 1) - 2 * nely])
        frcn = np.arange(2 * nely + 1, 2 * (nelx + 1) * (nely + 1), 8 * (nely + 1))
        bcSettings = {'fixedNodes': fixn,
                      'forceMagnitude': -1. / (nelx + 1.),
                      'forceNodes': frcn,
                      'dofsPerNode': 2}
        symMap = {'XAxis': {'isOn': False, 'midPt': 0.5 * nely * elemSize[1]},
                  'YAxis': {'isOn': True, 'midPt': 0.5 * nelx * elemSize[0]}}

    ## Tensile bar
    elif example == 6:
        exampleName = 'TensileBar'
        fn = np.union1d(np.arange(0, 2 * (nely + 1), 2), 1)
        # Find the mid of the degree of freedom in X
        midDofX = 2 * (nelx + 1) * (nely + 1) - nely
        bcSettings = {'fixedNodes': fn,
                      'forceMagnitude': 1.,
                      'forceNodes': np.arange(midDofX - 6, midDofX + 6, 2),
                      'dofsPerNode': 2}
        symMap = {'XAxis': {'isOn': False, 'midPt': 0.5 * nely * elemSize[1]},
                  'YAxis': {'isOn': False, 'midPt': 0.5 * nelx * elemSize[0]}}

    ## Full right cantilever
    elif example == 7:
        exampleName = 'fullRightCantilever'
        forceNodes = np.arange(2 * (nelx + 1) * (nely + 1) - 2 * nely + 1,
                               2 * (nelx + 1) * (nely + 1), 2)
        bcSettings = {'fixedNodes': np.arange(0, 2 * (nely + 1), 1),
                      'forceMagnitude': -100.,
                      'forceNodes': forceNodes,
                      'dofsPerNode': 2} # multiplyy C by 1000 to replicate
        symMap = {'XAxis': {'isOn': True, 'midPt': 0.5 * nely * elemSize[1]},
                  'YAxis': {'isOn': False, 'midPt': 0.5 * nelx * elemSize[0]}}

    ## Torsion
    elif example == 8:
        exampleName = 'Torsion'
        forceFile = './Mesh/Torsion/force.txt'
        fixedFile = './Mesh/Torsion/fixed.txt'
        nodeXYFile = './Mesh/Torsion/nodeXY.txt'
        elemNodesFile = './Mesh/Torsion/elemNodes.txt'
        bcSettings = {'forceFile': forceFile,
                      'fixedFile': fixedFile,
                      'elemNodesFile': elemNodesFile,
                      'nodeXYFile': nodeXYFile,
                      'dofsPerNode': 2}
        symMap = {'XAxis': {'isOn': True, 'midPt': 50},
                  'YAxis': {'isOn': True, 'midPt': 50}}

    ## L-Bracket
    elif example == 9:
        exampleName = 'LBracket'
        forceFile = './Mesh/LBracket/force.txt'
        fixedFile = './Mesh/LBracket/fixed.txt'
        nodeXYFile = './Mesh/LBracket/nodeXY.txt'
        elemNodesFile = './Mesh/LBracket/elemNodes.txt'
        bcSettings = {'forceFile': forceFile,
                      'fixedFile': fixedFile,
                      'elemNodesFile': elemNodesFile,
                      'nodeXYFile': nodeXYFile,
                      'dofsPerNode': 2}
        symMap = {'XAxis': {'isOn': False, 'midPt': 5},
                  'YAxis': {'isOn': False, 'midPt': 5}}

    ## Mid-load MBB
    elif example == 10:
        exampleName = 'midLoadMBB'
        forceFile = './Mesh/midLoadMBB/force.txt'
        fixedFile = './Mesh/midLoadMBB/fixed.txt'
        nodeXYFile = './Mesh/midLoadMBB/nodeXY.txt'
        elemNodesFile = './Mesh/midLoadMBB/elemNodes.txt'
        bcSettings = {'forceFile': forceFile,
                      'fixedFile': fixedFile,
                      'elemNodesFile': elemNodesFile,
                      'nodeXYFile': nodeXYFile,
                      'dofsPerNode': 2}
        symMap = {'XAxis': {'isOn': False, 'midPt': 5},
                  'YAxis': {'isOn': True, 'midPt': 50}}

    ## Mid-force cantilever with symmetry
    elif example == 11:
        exampleName = 'TipCantilever_Ed'
        bcSettings = {'fixedNodes': np.arange(0, 2 * (nely + 1), 1),
                      'forceMagnitude': -1.,
                      #'forceNodes': (nelx + 1) * (nely + 1) - 1,      # Localized/Concentrated force (45,15)
                      'forceNodes': (nelx + 1) * (nely + 1) + 30,      # Localized/Concentrated force (90,30)
                      #'forceNodes': [703, 735],
                      'dofsPerNode': 2}

        symMap = {'XAxis': {'isOn': False, 'midPt': 0.5 * nely * elemSize[1]},
                  'YAxis': {'isOn': True, 'midPt': 0.5 * nelx * elemSize[0]}}       # True, vf = 0.45, 1500 iter

    ## Cantilever with symetry (distributted force) - Bridge (1)
    elif example == 12:
        exampleName = 'TipCantilever_Ed'
        bcSettings = {'fixedNodes': np.arange(0, 2 * (nely + 1), 1),
                      'forceMagnitude': -1./45,
                      'forceNodes': np.arange(6 * (nely+1) - 1, (nelx + 2) * (nely + 2), 2 * (nely + 1)),  # distributted
                      'dofsPerNode': 2}

        symMap = {'XAxis': {'isOn': False, 'midPt': 0.5 * nely * elemSize[1]},
                  'YAxis': {'isOn': True, 'midPt': 0.5 * nelx * elemSize[0]}}       # True, vf = 0.65, 1500 iter

    ## Cantilever with symetry (distributted force) - Bridge (1)
    elif example == 13:
        #step = 2 * (nely + 1)
        #dyn_step = [step, 2 * step, 3 * step]
        #force_nodes = np.arange(6 * (nely + 1) - 1, (nelx + 1) * (nely + 1), 2 * (nely + 1))
        exampleName = 'TipCantilever_Ed'
        bcSettings = {'fixedNodes': np.arange(0, 2 * (nely + 1), 1),
                      'forceMagnitude': -1.,
                      #'forceNodes': np.arange(6 * (nely + 1) - 1, (nelx + 1) * (nely + 1), 2 * (nely + 1)),   # distributted 0
                      #'forceNodes': [95, 159, 255, 351, 511, 735],  # distributted 1
                      'forceNodes': [95, 287, 447, 575, 671, 735],  # distributted 2
                      'dofsPerNode': 2}

        symMap = {'XAxis': {'isOn': False, 'midPt': 0.5 * nely * elemSize[1]},
                  'YAxis': {'isOn': True, 'midPt': 0.5 * nelx * elemSize[0]}}       # True, vf = 0.65, 1500 iter, 0.35

    ## Cantilever with symmetry (distributted force) - Bridge - STUDY CASE
    elif example == 14:
        exampleName = 'TipCantilever_Ed'
        fixed_nodes_1 = np.arange(1, (nelx + 1) * (nely + 1), nely + 1)
        fixed_nodes_2 = np.arange(0, (nelx + 1) * (nely + 1), nely + 1)
        fixed_nodes = np.hstack((fixed_nodes_1, fixed_nodes_2))
        print(fixed_nodes)
        bcSettings = {'fixedNodes': fixed_nodes,
                      #'forceMagnitude': 1400000.,  # 10,3
                      'forceMagnitude': 280000.,    # 50,15
                      #'forceNodes': np.arange(26 * (nely+1) - 1, (nelx + 2) * (nely + 2), 2 * (nely + 1)),              # 90,30 elements
                      #'forceNodes': np.arange(2*(nely + 1)-1, 2 * (nelx + 1) * (nely + 1), 2 * (nely + 1)),             # 18,6
                      #'forceNodes': np.arange(8 * (nely + 1) - 1, 2 * (nelx + 1) * (nely + 1), 2 * (nely + 1)),          # 10,3
                      'forceNodes': np.arange(32 * (nely + 1) - 1, 2 * (nelx + 1) * (nely + 1), 2 * (nely + 1)),          # 50,15
                                                                               # 0 in x, 1 in y, 2 in x, 3, in y
                      #'forceNodes': 1,
                      'dofsPerNode': 2}

        symMap = {'XAxis': {'isOn': False, 'midPt': 0.5 * nely * elemSize[1]},
                  'YAxis': {'isOn': True, 'midPt': 0.5 * nelx * elemSize[0]}}       # True, vf = 0.65, 1500 iter
    else:
        exampleName = 'None'
        bcSettings = 'None'
        symMap = 'None'

    return exampleName, bcSettings, symMap
