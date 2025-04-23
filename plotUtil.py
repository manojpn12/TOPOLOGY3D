import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plotConvergence(convg):
    x = np.array(convg['epoch'])

    for key in convg:
        if key == 'epoch':
            # Epoch is x axis for all plots
            continue

        plt.figure()
        y = np.array(convg[key])
        plt.semilogy(x, y, label=str(key))
        plt.xlabel('Iterations')
        plt.ylabel(str(key))
        plt.grid('True')
        plt.legend()