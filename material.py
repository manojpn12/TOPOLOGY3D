import numpy as np


class Material:
    def __init__(self, matProp):
        self.matProp = matProp
        E, nu = matProp['Emax'], matProp['nu']

        # 3D elasticity stiffness matrix (6x6 Voigt notation)
        factor = E / ((1 + nu) * (1 - 2 * nu))
        self.C = factor * np.array([
            [1 - nu, nu, nu, 0, 0, 0],
            [nu, 1 - nu, nu, 0, 0, 0],
            [nu, nu, 1 - nu, 0, 0, 0],
            [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
            [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
            [0, 0, 0, 0, 0, (1 - 2 * nu) / 2]
        ])

    def computeSIMP_Interpolation(self, rho, penal):
        E = 0.001 * self.matProp['Emax'] + (0.999 * self.matProp['Emax']) * (rho + 0.01) ** penal
        return E


    def computeRAMP_Interpolation(self, rho, penal):
        E = 0.001 * self.matProp['Emax'] + (0.999 * self.matProp['Emax']) * (rho / (1. + penal * (1. - rho)))
        return E


    def getD0elemMatrix(self, mesh):
        if mesh.meshType == 'gridMesh':
            E = 1
            nu = self.matProp['nu']
            k = np.array([1/3 - nu/9, 1/6 + nu/6, -1/12 - nu/18, -1/6 + nu/3,
                        -1/12 + nu/18, -1/6 - nu/6, nu/9, 1/6 - nu/3])

            # all the elems have same base stiffness
            D0 = E / ((1 + nu) * (1 - 2 * nu)) * np.array(
                [[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                 [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                 [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                 [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                 [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                 [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                 [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                 [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])

            return D0
