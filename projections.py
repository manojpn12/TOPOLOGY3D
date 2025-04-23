import jax.numpy as jnp
import numpy as np
#from jax.ops import index, index_update
from jax.numpy import index_exp as index

## FOURIER LENGTH SCALE
def computeFourierMap(mesh, fourierMap):
    # Compute the map
    coordnMapSize = (mesh.ndim, fourierMap['numTerms'])
    freqSign = np.random.choice([-1., 1.], coordnMapSize)
    stdUniform = np.random.uniform(0., 1., coordnMapSize)
    wmin = 1. / (2 * fourierMap['maxRadius'])
    wmax = 1. / (2 * fourierMap['minRadius']) # w~1/R
    wu = wmin + (wmax - wmin) * stdUniform
    coordnMap = np.einsum('ij,ij->ij', freqSign, wu)

    print(f"Computed Fourier Map Shape: {coordnMap.shape}, Expected: {coordnMapSize}")
    return coordnMap

def applyFourierMap(xyz, fourierMap):
    """
    Apply Fourier mapping to input coordinates
    Returns an array with features representing the Fourier encoding of the input coordinates
    """
    if fourierMap['isOn']:
        try:
            # Debug the input shapes
            print(f"xyz shape: {xyz.shape}, Fourier map shape: {fourierMap['map'].shape}")
            
            # Ensure xyz has the right dimensionality for matrix multiplication
            xyz_ndim = fourierMap['map'].shape[0]
            if xyz.shape[1] != xyz_ndim:
                print(f"Dimension mismatch. Adjusting xyz from shape {xyz.shape}")
                if xyz.shape[1] > xyz_ndim:
                    # Truncate extra dimensions
                    xyz = xyz[:, :xyz_ndim]
                else:
                    # Pad with zeros
                    padding = np.zeros((xyz.shape[0], xyz_ndim - xyz.shape[1]))
                    xyz = np.hstack([xyz, padding])
                print(f"New xyz shape: {xyz.shape}")
            
            # Compute the projection (dot product)
            try:
                # Use standard matrix multiplication for stability
                projected = jnp.dot(xyz, fourierMap['map'])
                print(f"Projected shape: {projected.shape}")
                
                # Apply sinusoidal encoding
                cos_features = jnp.cos(2 * np.pi * projected)
                sin_features = jnp.sin(2 * np.pi * projected)
                
                # Concatenate sin and cos features
                result = jnp.concatenate([cos_features, sin_features], axis=1)
                print(f"Final encoded shape: {result.shape}")
                
                return result
                
            except Exception as e:
                print(f"Error in Fourier projection calculation: {e}")
                # Return a simple encoding as fallback
                num_terms = fourierMap.get('numTerms', 10)
                return jnp.ones((xyz.shape[0], 2 * num_terms)) * 0.1
                
        except Exception as e:
            print(f"Error in Fourier mapping: {e}")
            # Return original coordinates as fallback
            return xyz
    else:
        # Fourier mapping is disabled, return original coordinates
        return xyz


## DENSITY PROJECTION
def applyDensityProjection(x, densityProj):
    if densityProj['isOn']:
        b = densityProj['sharpness']
        nmr = np.tanh(0.5 * b) + jnp.tanh(b * (x - 0.5))
        x = 0.5 * nmr / np.tanh(0.5 * b)
        return x


## SYMMETRY
def applySymmetry(x, symMap):
    if symMap['YAxis']['isOn']:
        # original
        #xv = index_update(x[:, 0], index[:], symMap['YAxis']['midPt'] \
        #                   + jnp.abs(x[:, 0] - symMap['YAxis']['midPt']))
        xv = jnp.array(x[:, 0]).at[index[:]].set(symMap['YAxis']['midPt'] + jnp.abs(x[:, 0] - symMap['YAxis']['midPt']))
    else:
        xv = x[:, 0]

    if symMap['XAxis']['isOn']:
        # original
        #yv = index_update(x[:, 1], index[:], symMap['XAxis']['midPt'] \
        #                   + jnp.abs(x[:, 1] - symMap['XAxis']['midPt']))
        yv = jnp.array(x[:, 1]).at[index[:]].set(symMap['XAxis']['midPt'] + jnp.abs(x[:, 1] - symMap['XAxis']['midPt']))
    else:
        yv = x[:, 1]

    if 'ZAxis' in symMap and symMap['ZAxis']['isOn']:
        zv = jnp.array(x[:, 2]).at[index[:]].set(symMap['ZAxis']['midPt'] + jnp.abs(x[:, 2] - symMap['ZAxis']['midPt']))
    else:
        zv = x[:, 2]
        x_symmetric = jnp.stack((xv, yv, zv)).T

    return x_symmetric


def applyRotationalSymmetry(xyzCoordn, rotationalSymmetry):
    if not rotationalSymmetry['isOn']:
        return xyzCoordn  # Return unchanged if symmetry is off

    # Extract center coordinates
    cx, cy, cz = rotationalSymmetry['centerCoordn']
    sectorAngleRad = np.pi * rotationalSymmetry['sectorAngleDeg'] / 180.0  # Convert to radians

    axis = rotationalSymmetry.get('axis', 'XY')  # Default rotation in XY-plane

    if axis == 'XY':
        dx = xyzCoordn[:, 0] - cx
        dy = xyzCoordn[:, 1] - cy
        dz = xyzCoordn[:, 2]  # Z remains unchanged

        # Compute radius and angle in the XY-plane
        radius = jnp.sqrt(dx**2 + dy**2)
        angle = jnp.arctan2(dy, dx)

        # Apply sector constraint
        correctedAngle = jnp.remainder(angle, sectorAngleRad)

        # Compute new X, Y coordinates
        x_new = radius * jnp.cos(correctedAngle) + cx
        y_new = radius * jnp.sin(correctedAngle) + cy

        # Reconstruct new coordinates
        xyz_symmetric = jnp.stack((x_new, y_new, dz)).T

    elif axis == 'YZ':
        dx = xyzCoordn[:, 1] - cy
        dy = xyzCoordn[:, 2] - cz
        dz = xyzCoordn[:, 0]  # X remains unchanged

        # Compute radius and angle in the YZ-plane
        radius = jnp.sqrt(dx**2 + dy**2)
        angle = jnp.arctan2(dy, dx)

        # Apply sector constraint
        correctedAngle = jnp.remainder(angle, sectorAngleRad)

        # Compute new Y, Z coordinates
        y_new = radius * jnp.cos(correctedAngle) + cy
        z_new = radius * jnp.sin(correctedAngle) + cz

        # Reconstruct new coordinates
        xyz_symmetric = jnp.stack((dz, y_new, z_new)).T

    elif axis == 'XZ':
        dx = xyzCoordn[:, 0] - cx
        dy = xyzCoordn[:, 2] - cz
        dz = xyzCoordn[:, 1]  # Y remains unchanged

        # Compute radius and angle in the XZ-plane
        radius = jnp.sqrt(dx**2 + dy**2)
        angle = jnp.arctan2(dy, dx)

        # Apply sector constraint
        correctedAngle = jnp.remainder(angle, sectorAngleRad)

        # Compute new X, Z coordinates
        x_new = radius * jnp.cos(correctedAngle) + cx
        z_new = radius * jnp.sin(correctedAngle) + cz

        # Reconstruct new coordinates
        xyz_symmetric = jnp.stack((x_new, dz, z_new)).T

    else:
        raise ValueError(f"Invalid rotation axis '{axis}'. Choose from 'XY', 'YZ', or 'XZ'.")

    return xyz_symmetric


def applyExtrusion(xyz, extrusion):
    if extrusion['X']['isOn']:
        xv = xyz[:, 0] % extrusion['X']['delta']
    else:
        xv = xyz[:, 0]
    if extrusion['Y']['isOn']:
        yv = xyz[:, 1] % extrusion['Y']['delta']
    else:
        yv = xyz[:, 1]
    if extrusion['Z']['isOn']:
        zv = xyz[:, 2] % extrusion['Z']['delta']
    else:
        zv = xyz[:, 2]

    xyz_extruded = jnp.stack((xv, yv, zv)).T

    return xyz_extruded
