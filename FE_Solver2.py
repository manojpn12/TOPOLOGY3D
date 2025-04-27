import jax.numpy as jnp
import numpy as np
from jax import jit
import jax
from FE_templates2_val import getKMatrixGridMeshTemplates
import time


class JAXSolver:
    def __init__(self, mesh, material):
        self.mesh = mesh
        self.material = material
        self.Ktemplates = getKMatrixGridMeshTemplates(mesh.elemSize, 'structural')
        self.objectiveHandle = self.objective
        self.D0 = self.material.getD0elemMatrix(self.mesh)

    def objective(self, C):
        @jit
        def assembleK(C):
            # For 3D, each element has 8 nodes with 3 DOFs per node = 24 DOFs per element
            sK = jnp.zeros((self.mesh.numElems, 24, 24))
            for k in C:
                sK += jnp.einsum('e,jk->ejk', C[k], self.Ktemplates[k])
            K = jnp.zeros((self.mesh.ndof, self.mesh.ndof))
            K = K.at[self.mesh.nodeIdx].add(sK.flatten())
            K+= 5e-3 *jnp.eye(self.mesh.ndof)  # Add small value to diagonal for numerical stability

            return K

        @jit
        def solve(K):
            # Eliminate fixed dofs for solving sys of eqns
            u_free = jax.scipy.linalg.solve(K[self.mesh.bc['free'], :][:, self.mesh.bc['free']],
                                            self.mesh.bc['force'][self.mesh.bc['free']], assume_a='pos',
                                            check_finite=False)
            u = jnp.zeros(self.mesh.ndof)
            u = u.at[self.mesh.bc['free']].set(u_free.reshape(-1))
            return u

        @jit
        def computeCompliance(K, u):
            J = jnp.dot(self.mesh.bc['force'].reshape(-1).T, u)
            return J

        # Add timing for K assembly
        start_assembly = time.perf_counter()
        K = assembleK(C)
        jax.debug.print("🔎 NaN check: Any NaN in global K matrix? {}", jnp.any(jnp.isnan(K)))
        jax.debug.print("🔎 NaN check: Any Inf in global K matrix? {}", jnp.any(jnp.isinf(K)))
        min_val = jnp.min(jnp.abs(K))
        jax.debug.print("🔎 Minimum absolute value in K: {}", min_val)
        end_assembly = time.perf_counter()
        
        # Add timing for solving system
        start_solve = time.perf_counter()
        K_free = K[self.mesh.bc['free'], :][:, self.mesh.bc['free']]
        jax.debug.print("🔎 Condition number of K_free: {}", jnp.linalg.cond(K_free))
        u = solve(K)

        jax.debug.print("🔎 NaN check: Any NaN in global K matrix? {}", jnp.any(jnp.isnan(u)))
        jax.debug.print("🔎 NaN check: Any Inf in global K matrix? {}", jnp.any(jnp.isinf(u)))
        end_solve = time.perf_counter()
        
        # Calculate compliance
        J = computeCompliance(K, u)
        
        # Print timing information (optional - can be commented out for production)
        print(f"Stiffness assembly time: {end_assembly - start_assembly:.4f} seconds")
        print(f"Equation solving time: {end_solve - start_solve:.4f} seconds")
        
        return J