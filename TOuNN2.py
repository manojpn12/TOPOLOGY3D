import numpy as np
import jax.numpy as jnp
import jax
from jax import jit
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from FE_Solver2 import JAXSolver
from network import TopNet
from projections import applyFourierMap, applySymmetry, applyRotationalSymmetry, applyExtrusion
from jax.example_libraries import optimizers
from materialCoeffs import microStrs
import pickle
import time
import pandas as pd


class TOuNN:
    def __init__(self, exampleName, mesh, material, nnSettings, symMap, fourierMap, rotationalSymmetry, extrusion):
        self.exampleName = exampleName
        self.FE = JAXSolver(mesh, material)
        self.xyz = self.FE.mesh.elemCenters
        self.nnSettings = nnSettings
        self.fourierMap = fourierMap
        
        # Track timing information
        self.timers = {
            'getCMatrix': 0.0,
            'forward_pass': 0.0,
            'gradient_computation': 0.0,
            'parameter_update': 0.0,
            'visualization': 0.0,
            'optimization_phases': [],
            'total_optimization': 0.0
        }
        
        # Update input dimension based on Fourier mapping
        if fourierMap['isOn']:
            nnSettings['inputDim'] = 2 * fourierMap['numTerms']
        else:
            nnSettings['inputDim'] = self.FE.mesh.ndim
        
        # Add target volume fraction to nnSettings (will be updated in optimizeDesign)
        nnSettings['target_vf'] = 0.7 # Default value, will be overridden
        
        # Initialize the neural network
        self.topNet = TopNet(nnSettings)
        
        self.symMap = symMap
        self.mstrData = microStrs
        self.rotationalSymmetry = rotationalSymmetry
        self.extrusion = extrusion

        #fixed_nodes = self.FE.mesh.bc['fixed'] // self.FE.mesh.bcSettings['dofsPerNode']
        #fixed_nodes = jnp.array(fixed_nodes)  # ensure it's a JAX array
        #self.maskFixedElems = jnp.any(jnp.isin(self.FE.mesh.elemNodes, fixed_nodes), axis=1)
        #print(f"Fixed-connected elements: {jnp.sum(self.maskFixedElems)} / {self.FE.mesh.numElems}")


    def getCfromCPolynomial(self, vfracPow, mstr):
        components = ['00', '11', '22', '01', '02', '12', '44', '55', '66']
        C = {}
        for c in components:
            C[c] = jnp.zeros(self.FE.mesh.numElems)
        for c in components:            
            for pw in range(mstr['order'] + 1):
                C[c] = C[c].at[:].add(mstr[c][str(pw)] * vfracPow[str(pw)])
        return C


    def getCMatrix(self, mstrType, density, penal=1.0):
        components = ['00', '11', '22', '01', '02', '12', '44', '55', '66']
        
        # Step 1: Prepare powers of density
        vfracPow = {}
        for pw in range(self.mstrData['cube']['order'] + 1):
            vfracPow[str(pw)] = density ** pw

        # Step 2: Prepare empty C matrices
        C = {c: jnp.zeros(self.FE.mesh.numElems) for c in components}

        # Step 3: Initialize debug log
        debug_log = []

        # Step 4: Prepare for optional debugging if not in autodiff mode
        if not isinstance(density, jax.core.Tracer):
            C_raw = {c: jnp.zeros(self.FE.mesh.numElems) for c in components}
            idx = jnp.array(np.arange(min(3, self.FE.mesh.numElems)))
            print("=== Raw C (pre-clip) and Penalized C for sample elements ===")
            
        # Step 5: Start main loops
        for mstrCtr, mstrName in enumerate(self.mstrData):
            mstr = self.mstrData[mstrName]
            Cmstr = self.getCfromCPolynomial(vfracPow, mstr)
            mstrPenal = mstrType[:, mstrCtr] ** penal

            # Add mstrType stats
            mstrType_min = jnp.min(mstrType[:, mstrCtr])
            mstrType_max = jnp.max(mstrType[:, mstrCtr])
            mstrType_mean = jnp.mean(mstrType[:, mstrCtr])

            for c in components:
                # Values BEFORE clipping
                cmstr_max = jnp.max(Cmstr[c])
                cmstr_min = jnp.min(Cmstr[c])
                cmstr_nan = jnp.any(jnp.isnan(Cmstr[c]))

                stabilized_C = jnp.clip(Cmstr[c], 1e-2, None)

                # Values AFTER clipping
                stabilized_max = jnp.max(stabilized_C)
                stabilized_min = jnp.min(stabilized_C)
                stabilized_nan = jnp.any(jnp.isnan(stabilized_C))

                # Optional raw value collection
                if not isinstance(density, jax.core.Tracer):
                    C_raw[c] = C_raw[c].at[:].add(jnp.einsum('i,i->i', mstrType[:, mstrCtr], stabilized_C))

                # Apply penalization and add to C
                C[c] = C[c].at[:].add(jnp.einsum('i,i->i', mstrPenal, stabilized_C))

                # Values AFTER penalization
                penalized_max = jnp.max(C[c])
                penalized_min = jnp.min(C[c])
                penalized_nan = jnp.any(jnp.isnan(C[c]))

                # Step 6: Collect debug info
                debug_log.append({
                    'mstrName': mstrName,
                    'component': c,
                    'Cmstr_max': cmstr_max,
                    'Cmstr_min': cmstr_min,
                    'Cmstr_any_nan': cmstr_nan,
                    'stabilized_max': stabilized_max,
                    'stabilized_min': stabilized_min,
                    'stabilized_any_nan': stabilized_nan,
                    'C_max_after_penalization': penalized_max,
                    'C_min_after_penalization': penalized_min,
                    'C_any_nan_after_penalization': penalized_nan,
                    'density_min': jnp.min(density),
                    'density_max': jnp.max(density),
                    'density_mean': jnp.mean(density),
                    'mstrType_min': mstrType_min,
                    'mstrType_max': mstrType_max,
                    'mstrType_mean': mstrType_mean,
                })

        # Step 7: After loops, print and save debug info
        if not isinstance(density, jax.core.Tracer):
            for k in components:
                raw_vals = C_raw[k][idx]
                penalized_vals = C[k][idx]
                print(f" C[{k}]: raw={raw_vals}, penalized={penalized_vals}, penal_factor={penal}")

            # Save debug log into CSV
            df_debug = pd.DataFrame(debug_log)
            df_debug.to_csv("debug_log_cmatrix.csv", index=False)
            print("✅ Debug log saved to 'debug_log_cmatrix.csv'")
        # C_all = jnp.concatenate([C[k] for k in components])
        # jax.debug.print("✅ Final NaN check across all C: {}", jnp.any(jnp.isnan(C_all)))
        return C



    def optimizeDesign(self, optParams, savedNet):
        """
        Optimize the design for 3D structures using SIMP model
        """
        total_start_time = time.time()
        phase_times = []
        print("Starting topology optimization for 3D structures...")
        
        # Get and set target volume fraction from optimization parameters
        target_vf = optParams.get('desiredVolumeFraction', 0.7)
        self.topNet.target_vf = target_vf
        print(f"Target volume fraction set to: {target_vf}")
        
        # Initialize tracking variables
        convgHistory = {'epoch': [], 'vf': [], 'J': []}
        
        # Apply projections to element centers
        xyzS = applySymmetry(self.xyz, self.symMap)
        xyzE = applyExtrusion(xyzS, self.extrusion)
        xyzR = applyRotationalSymmetry(xyzE, self.rotationalSymmetry)

        # Apply Fourier mapping if enabled
        if self.fourierMap['isOn']:
            xyzF = applyFourierMap(xyzR, self.fourierMap)
            print(f"Fourier mapping applied, shape: {xyzF.shape}")
        else:
            xyzF = xyzR
            print(f"No Fourier mapping, shape: {xyzF.shape}")

        # Handle dimension mismatch if needed
        expected_input_dim = self.nnSettings['inputDim']
        if xyzF.shape[1] != expected_input_dim:
            print(f"Input dimension mismatch: got {xyzF.shape[1]}, expected {expected_input_dim}")
            
            if xyzF.shape[1] > expected_input_dim:
                # If we have too many features, take only what we need
                print(f"Truncating input features to {expected_input_dim}")
                xyzF = xyzF[:, :expected_input_dim]
            else:
                # If we have too few features, pad with zeros
                padding_needed = expected_input_dim - xyzF.shape[1]
                print(f"Padding input with {padding_needed} zero columns")
                zero_padding = np.zeros((xyzF.shape[0], padding_needed))
                xyzF = np.hstack([xyzF, zero_padding])

        # Clip values to prevent numerical instability
        xyzF = jnp.clip(xyzF, -10, 10)
        
        # Initialize ADAM optimizer
        opt_init, opt_update, get_params = optimizers.adam(optParams['learningRate'])
        opt_state = opt_init(self.topNet.wts)
        
        # Try to load saved network if available
        if savedNet['isAvailable']:
            try:
                saved_params = pickle.load(open(savedNet['file'], "rb"))
                opt_state = opt_init(saved_params)
                print("Successfully loaded saved network weights")
            except Exception as e:
                print(f"Error loading saved network: {e}")
        
        # Ensure penal starts at a reasonable value
        penal = 1.0
        
        # Closure function for optimization
        def closure(params):
            try:
                # Forward pass
                start_forward = time.time()
                mstrType, density = self.topNet.forward(params, xyzF)
                end_forward = time.time()
                self.timers['forward_pass'] += (end_forward - start_forward)
                
                #Normalize density
                current_vf = jnp.mean(density)
                error = target_vf - current_vf
                adjustment_rate = 0.2  # Tune this parameter (0.05-0.2)
                density = jnp.clip(density * (1 + adjustment_rate * error), 0.05, 0.99)
                jax.debug.print("🛡️ Clipped density min: {}, max: {}, mean: {}", jnp.min(density), jnp.max(density), jnp.mean(density))
                #if hasattr(self, 'maskFixedElems'):
                    #density = jnp.where(self.maskFixedElems, jnp.maximum(density, 0.2), density)

                
                # Calculate volume constraint
                current_vf = jnp.mean(density)
                volCons = current_vf / target_vf - 1.0
                
                # Get C matrix with appropriate penalization
                start_C = time.perf_counter()
                C = self.getCMatrix(mstrType, density, penal)

                end_C = time.perf_counter()
                self.timers['getCMatrix'] += (end_C - start_C)
                for key in C.keys():
                    jax.debug.print("🔎 NaN check: C[{}] has NaNs: {}", key, jnp.any(jnp.isnan(C[key])))
                
                # Calculate compliance with fallback for NaN
                try:
                    J = self.FE.objectiveHandle(C)
                    if jnp.isnan(J) or jnp.isinf(J):
                        print("Warning: Invalid compliance detected, using fallback value")
                        J = jnp.array(1e6, dtype=jnp.float32)
                except Exception as e:
                    print(f"Error in compliance calculation: {e}")
                    J = jnp.array(1e6, dtype=jnp.float32)
                
                # Apply volume constraint penalty using log-barrier or penalty method
                if optParams['lossMethod']['type'] == 'penalty':
                    alpha = optParams['lossMethod']['alpha0'] * (1.2 ** self.epoch)
                    if volCons > 0:  # We're over the volume constraint
                        # Apply stronger penalty when over the target volume
                        loss = J + 3.0 * alpha * jnp.square(volCons)
                    else:
                        # Less aggressive when under the target
                        loss = J + alpha * jnp.square(volCons)
                elif optParams['lossMethod']['type'] == 'logBarrier' and volCons < 0:
                    t = optParams['lossMethod']['t0'] * (optParams['lossMethod']['mu'] ** self.epoch)
                    psi = -jnp.log(-volCons) / t
                    loss = J + psi
                else:
                    # Default fallback with stronger enforcement
                    if volCons > 0:  # Over the constraint
                        loss = J + 3000.0 * jnp.square(volCons)
                    else:
                        loss = J + 1000.0 * jnp.square(volCons)

                return loss
            
                    
            except Exception as e:
                print(f"Error in optimization step: {e}")
                import traceback
                traceback.print_exc()
                return jnp.array(1e5, dtype=jnp.float32)
        
        # Multi-phase optimization with gradual penalty increase
        phases = [
            {'max_epochs': 150, 'penal': 2.0, 'lr': optParams['learningRate']},
            {'max_epochs': 150, 'penal': 4.0, 'lr': optParams['learningRate'] * 0.5},
            {'max_epochs': 150, 'penal': 6.0, 'lr': optParams['learningRate'] * 0.25},
            {'max_epochs': 150, 'penal': 8.0, 'lr': optParams['learningRate'] * 0.1},
        ]
        
        # Calculate total epochs
        total_epochs = sum(phase['max_epochs'] for phase in phases)
        print(f"Total optimization epochs: {total_epochs}")
        
        # Initialize metrics
        J = 1e10
        vf = target_vf
        
        # Run optimization phases
        epoch_counter = 0
        for phase_idx, phase in enumerate(phases):
            start_phase = time.time()
            print(f"\n--- Starting optimization phase {phase_idx+1}/{len(phases)} ---")
            print(f"Penalty: {phase['penal']}, Learning rate: {phase['lr']}")
            
            # Update current penalization factor
            penal = phase['penal']

            
            # Update optimizer learning rate
            opt_state = opt_init(get_params(opt_state))  # Reinitialize with current params
            opt_update_phase = optimizers.adam(phase['lr'])[1]  # Get updated optimizer
            
            # Run epochs for this phase
            for i in range(phase['max_epochs']):
                self.epoch = epoch_counter

                
                try:
                    # Compute gradients
                    start_grad = time.time()
                    params = get_params(opt_state)
                    grads = jax.grad(closure)(params)
                    end_grad = time.time()
                    self.timers['gradient_computation'] += (end_grad - start_grad)
                    
                    # Check for invalid gradients
                    has_invalid = False
                    for g in jax.tree_util.tree_leaves(grads):
                        if jnp.any(jnp.isnan(g)) or jnp.any(jnp.isinf(g)):
                            has_invalid = True
                            break
                    
                    if has_invalid:
                        print("Warning: Invalid gradients detected, using zeros")
                        grads = jax.tree_map(lambda g: jnp.zeros_like(g), grads)
                    
                    # Apply gradient clipping if configured
                    if optParams.get('gradclip', {}).get('isOn', False):
                        clip_value = optParams['gradclip'].get('thresh', 1.0)
                        grads = optimizers.clip_grads(grads, clip_value)
                    
                    # Update parameters
                    start_update = time.time()
                    opt_state = opt_update_phase(self.epoch, grads, opt_state)
                    end_update = time.time()
                    self.timers['parameter_update'] += (end_update - start_update)
                    
                    # Compute current metrics
                    mstrType, density = self.topNet.forward(get_params(opt_state), xyzF)
                    density = jnp.clip(jnp.nan_to_num(density, 0.01, 0.99))
                    if self.epoch % 5 == 0:
                        print(f"Epoch {self.epoch} density stats:")
                        print("  min:", jnp.min(density))
                        print("  max:", jnp.max(density))
                        print("  mean:", jnp.mean(density))
                    C = self.getCMatrix(mstrType, density, penal)
                    if self.epoch % 5 == 0:
                        for key in ['00', '11', '22', '01', '02', '12']:
                            print(f"[Epoch {self.epoch}] C[{key}] min: {jnp.min(C[key])}, max: {jnp.max(C[key])}, mean: {jnp.mean(C[key])}")
                    # ⬇️ Add this block to catch bad stiffness
                    for key, val in C.items():
                        if jnp.any(jnp.isnan(val)) or jnp.any(jnp.isinf(val)):
                            print(f"[Epoch {self.epoch}] NaN/Inf detected in C[{key}]!")
                        elif jnp.min(val) < 1e-6:
                            print(f"[Epoch {self.epoch}] Low values in C[{key}] — min:", jnp.min(val))
                    
                    try:
                        J = float(self.FE.objectiveHandle(C))
                        # Handle NaN in J
                        if np.isnan(J) or np.isinf(J):
                            J = 1e6
                    except Exception as e:
                        print(f"Error computing compliance: {e}")
                        J = 1e6
                        
                    vf = float(jnp.mean(density))
                    
                    # Store metrics and update plot every 5 epochs
                    if self.epoch % 5 == 0:
                        # Format status message
                        status = f'epoch: {self.epoch}/{total_epochs}, J: {J:.2E}, vf: {vf:.2F}'
                        print(status)
                        
                        # Store metrics
                        convgHistory['epoch'].append(self.epoch)
                        convgHistory['J'].append(J)
                        convgHistory['vf'].append(vf)
                        
                        # Track microstructure distribution
                        if len(mstrType.shape) > 1 and mstrType.shape[1] > 1:
                            mstr_probs = np.array(mstrType)
                            dominant_type = np.argmax(mstr_probs, axis=1)
                            type_counts = np.bincount(dominant_type, minlength=3)
                            
                            # Output microstructure distribution
                            type_names = ['Cube', 'Xbox', 'XPlusBox']
                            dist_str = ", ".join([f"{type_names[i]}: {type_counts[i]/np.sum(type_counts)*100:.1f}%" 
                                                for i in range(len(type_names))])
                            print(f"Microstructure distribution: {dist_str}")
                            
                            # Apply diversity perturbation if needed
                            max_type_ratio = np.max(type_counts) / np.sum(type_counts)
                            if max_type_ratio > 0.7:  # If one type dominates too much
                                print("Applying diversity perturbation")
                                
                                # Get current parameters
                                params = get_params(opt_state)
                                new_params = []
                                
                                # Calculate dominant type index
                                dominant_idx = np.argmax(type_counts)
                                
                                # Target the output layer to change microstructure selection
                                for i, p in enumerate(params):
                                    if i == len(params) - 2:  # Output layer weights
                                        # Apply targeted noise to promote diversity
                                        noise = np.zeros_like(p)
                                        
                                        # Add negative noise to dominant type (more negative for more dominance)
                                        noise[:, dominant_idx] = -0.08 * max_type_ratio
                                        
                                        # Add positive noise to other types
                                        remaining_weight = 0.08 * (1 - max_type_ratio)
                                        for j in range(3):
                                            if j != dominant_idx:
                                                noise[:, j] = remaining_weight
                                                
                                        new_params.append(p + noise)
                                    elif i == len(params) - 1:  # Output layer bias
                                        # Also modify bias terms
                                        noise = np.zeros_like(p)
                                        noise[dominant_idx] = -0.1
                                        for j in range(3):
                                            if j != dominant_idx:
                                                noise[j] = 0.05
                                        new_params.append(p + noise)
                                    else:
                                        # Very small noise for other layers
                                        noise = np.random.normal(0, 0.005, p.shape)
                                        new_params.append(p + noise)
                                
                                # Reinitialize optimizer with perturbed parameters
                                opt_state = opt_init(new_params)
                        
                        # Plot current field
                        try:
                            self.FE.mesh.plotFieldOnMesh(density, status)
                        except Exception as e:
                            print(f"Error plotting field: {e}")
                    
                except Exception as e:
                    print(f"Error in optimization iteration {self.epoch}: {e}")
                    import traceback
                    traceback.print_exc()
                
                epoch_counter += 1

            end_phase = time.time()
            phase_time = end_phase - start_phase
            phase_times.append(phase_time)
            self.timers['optimization_phases'].append(phase_time)
            print(f"Phase {phase_idx+1} time: {phase_time:.4f} seconds")
        
        # Update final weights
        self.topNet.wts = get_params(opt_state)

        # Save weights if requested
        if savedNet.get('isDump', False):
            try:
                pickle.dump(self.topNet.wts, open(savedNet['file'], "wb"))
                print(f"Successfully saved network weights to {savedNet['file']}")
            except Exception as e:
                print(f"Error saving network weights: {e}")

        # Print timing information for phases
        for i, time_taken in enumerate(phase_times):
            print(f"Phase {i+1}/{len(phases)} took {time_taken:.4f} seconds ({time_taken/sum(phase_times)*100:.1f}%)")
        
        # Calculate and store total optimization time
        total_end_time = time.time()
        self.timers['total_optimization'] = total_end_time - total_start_time
        print(f"Total optimization time: {self.timers['total_optimization']:.4f} seconds")
        
        # Print detailed timing breakdown
        print("\n--- Timing Breakdown ---")
        print(f"C-matrix calculations: {self.timers['getCMatrix']:.4f} seconds")
        print(f"Forward passes: {self.timers['forward_pass']:.4f} seconds")
        print(f"Gradient computations: {self.timers['gradient_computation']:.4f} seconds")
        print(f"Parameter updates: {self.timers['parameter_update']:.4f} seconds")
        print("--------------------------------------------------")
        
        return convgHistory

    def plotCompositeTopology(self, res):
        """
        Visualization for 3D topology optimization results
        """
        start_viz = time.time()
        try:
            # Generate samples at higher resolution if specified
            xyz = self.FE.mesh.generatePoints(res)
            xyzS = applySymmetry(xyz, self.symMap)
            xyzE = applyExtrusion(xyzS, self.extrusion)
            xyzR = applyRotationalSymmetry(xyzE, self.rotationalSymmetry)

            if self.fourierMap['isOn']:
                xyzF = applyFourierMap(xyzR, self.fourierMap)
            else:
                xyzF = xyzR
                
            # Forward pass to get microstructure types and densities
            mstrType, density = self.topNet.forward(self.topNet.wts, xyzF)
            
            # Handle NaN values
            density = np.array(jnp.nan_to_num(density, nan=0.1))
            
            # Ensure mstrType has the correct shape
            if len(np.array(mstrType).shape) == 1:
                print(f"Reshaping mstrType from 1D array to 2D array with 3 columns")
                mstrType_reshaped = np.zeros((len(mstrType), 3))
                mstrType_reshaped[:, 0] = 1
                mstrType = mstrType_reshaped
            else:
                mstrType = np.array(jnp.nan_to_num(mstrType, nan=1.0/3.0))
            
            # Apply stronger contrast to microstructure selection
            temperature = 0.05  # Very low temperature for sharp contrast
            mstrType_exp = np.exp(mstrType / temperature)
            mstrType = mstrType_exp / np.sum(mstrType_exp, axis=1, keepdims=True)
            
            # Get discrete microstructure selection
            mstr_selection = np.argmax(mstrType, axis=1)

            # Apply density threshold to filter out void elements
            threshold = 0.3
            print(f"Applying density threshold of {threshold}, original element count: {len(density)}")
            density_mask = density > threshold
            xyz_filtered = xyz[density_mask]
            mstr_selection_filtered = mstr_selection[density_mask]
            density_filtered = density[density_mask]
            print(f"After thresholding: {np.sum(density_mask)} elements remain (approx. {np.sum(density_mask)/len(density)*100:.1f}% of original)")
            
            # Update variables to use filtered arrays
            xyz = xyz_filtered
            mstr_selection = mstr_selection_filtered
            density = density_filtered
            
            # Get unique microstructure types
            unique_types = np.unique(mstr_selection)
            
            # Print statistics
            print(f"Density stats - min: {np.min(density)}, max: {np.max(density)}, mean: {np.mean(density)}")
            print(f"Unique microstructure types selected: {unique_types}")
            
            # Create visualization with multiple views
            fig = plt.figure(figsize=(15, 5))
            plt.suptitle('3D Multiscale Topology Optimization Results', fontsize=16)
            
            # 1. Overall structure visualization
            ax1 = fig.add_subplot(131, projection='3d')
            
            # Microstructure type colors
            colors = ['red', 'green', 'blue']
            microstructure_names = ['Cube', 'Xbox', 'XPlusBox']
            
            # Create scatter plot with colors for microstructure types and size for density
            legend_handles = []
            
            for i, mtype in enumerate(unique_types):
                idx = mstr_selection == mtype
                if np.any(idx):  # Only plot if there are points of this type
                    scatter = ax1.scatter(
                        xyz[idx, 0], xyz[idx, 1], xyz[idx, 2],
                        c=colors[mtype % len(colors)],
                        s=20 + 80 * density[idx],  # Size represents density
                        alpha=0.7,
                        label=f"{microstructure_names[mtype]}"
                    )
                    legend_handles.append(scatter)
                    
            # Add legend if we have handles
            if legend_handles:
                ax1.legend(handles=legend_handles)
            else:
                # Add a dummy scatter point and legend for visualization
                dummy = ax1.scatter([-1], [-1], [-1], c='none', alpha=0)
                ax1.legend([dummy], ["No microstructures found"])
                
            ax1.set_title('Optimized Structure')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            
            # 2. Density distribution
            ax2 = fig.add_subplot(132, projection='3d')
            scatter = ax2.scatter(
                xyz[:, 0], xyz[:, 1], xyz[:, 2],
                c=density,
                cmap='viridis',
                s=30,
                alpha=0.7
            )
            fig.colorbar(scatter, ax=ax2, label='Density')
            ax2.set_title('Density Distribution')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            
            # 3. Microstructure type distribution chart
            ax3 = fig.add_subplot(133)
            type_counts = np.bincount(mstr_selection, minlength=3)
            type_percent = 100 * type_counts / len(mstr_selection)
            bars = ax3.bar(microstructure_names, type_percent, color=colors)
            
            # Add percentage labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    ax3.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 1,
                        f'{height:.1f}%',
                        ha='center', va='bottom'
                    )
                    
            ax3.set_title('Microstructure Distribution')
            ax3.set_ylabel('Percentage (%)')
            ax3.set_ylim(0, 100)
            
            plt.tight_layout()
            plt.show()

            # Store element data to a text file
            results_data = []
            for i in range(len(xyz)):
                # Store density, microstructure type, and x,y,z coordinates
                results_data.append([
                    float(density[i] * 100),  # Convert to percentage (0-100)
                    int(mstr_selection[i]),   # Microstructure type index
                    float(xyz[i, 0]),         # X coordinate
                    float(xyz[i, 1]),         # Y coordinate
                    float(xyz[i, 2])          # Z coordinate
                ])

            # Save the data to a text file
            try:
                output_filename = f'predictions_3d_{self.exampleName}_vf_{self.topNet.target_vf:.1f}.txt'
                with open(output_filename, 'w') as file:
                    for row in results_data:
                        file.write('\t'.join(map(str, row)) + '\n')
                print(f"Saved microstructure data to {output_filename}")
            except Exception as e:
                print(f"Error saving data to file: {e}")

            # Add another visualization showing cross-sections if needed
            try:
                # Create sliced views for better visualization of internal structure
                fig2 = plt.figure(figsize=(15, 5))
                plt.suptitle('3D Structure Cross-Sections', fontsize=16)
                
                # Get coordinate ranges
                x_range = [np.min(xyz[:,0]), np.max(xyz[:,0])]
                y_range = [np.min(xyz[:,1]), np.max(xyz[:,1])]
                z_range = [np.min(xyz[:,2]), np.max(xyz[:,2])]
                
                # Create cross-sectional views at midpoints
                mid_x = (x_range[0] + x_range[1]) / 2
                mid_y = (y_range[0] + y_range[1]) / 2
                mid_z = (z_range[0] + z_range[1]) / 2
                
                # X-slice (YZ plane)
                ax_x = fig2.add_subplot(131)
                mask_x = np.abs(xyz[:,0] - mid_x) < (x_range[1] - x_range[0])/10  # 10% slice thickness
                ax_x.scatter(xyz[mask_x,1], xyz[mask_x,2], 
                            c=[colors[int(t)] for t in mstr_selection[mask_x]], 
                            s=50*density[mask_x], alpha=0.7)
                ax_x.set_title(f'YZ Plane (X={mid_x:.2f})')
                ax_x.set_xlabel('Y')
                ax_x.set_ylabel('Z')
                
                # Y-slice (XZ plane)
                ax_y = fig2.add_subplot(132)
                mask_y = np.abs(xyz[:,1] - mid_y) < (y_range[1] - y_range[0])/10
                ax_y.scatter(xyz[mask_y,0], xyz[mask_y,2], 
                            c=[colors[int(t)] for t in mstr_selection[mask_y]], 
                            s=50*density[mask_y], alpha=0.7)
                ax_y.set_title(f'XZ Plane (Y={mid_y:.2f})')
                ax_y.set_xlabel('X')
                ax_y.set_ylabel('Z')
                
                # Z-slice (XY plane)
                ax_z = fig2.add_subplot(133)
                mask_z = np.abs(xyz[:,2] - mid_z) < (z_range[1] - z_range[0])/10
                ax_z.scatter(xyz[mask_z,0], xyz[mask_z,1], 
                            c=[colors[int(t)] for t in mstr_selection[mask_z]], 
                            s=50*density[mask_z], alpha=0.7)
                ax_z.set_title(f'XY Plane (Z={mid_z:.2f})')
                ax_z.set_xlabel('X')
                ax_z.set_ylabel('Y')
                
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Error creating cross-section plots: {e}")
            
            # Update visualization timer
            end_viz = time.time()
            self.timers['visualization'] += (end_viz - start_viz)
            print(f"Topology Visualization time: {end_viz - start_viz:.4f} seconds")
            
        except Exception as e:
            print(f"Error in visualization: {e}")
            import traceback
            traceback.print_exc()