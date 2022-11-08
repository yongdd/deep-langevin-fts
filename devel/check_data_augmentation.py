import os
import sys
import time
import numpy as np
from langevinfts import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deep_langevin_fts import *

# -------------- simulation parameters ------------

#os.environ["CUDA_VISIBLE_DEVICES"]= "0"
verbose_level = 1  # 1 : print at each langevin step.
                   # 2 : print at each saddle point iteration.
                 
# Simulation Grids and Lengths
nx = [64,64,64]      
lx = [7.31,7.31,7.31]
lx = [lx[0],0.8*lx[0],1.2*lx[0]]

# Polymer Chain
n_segment = 90
f = 0.4
chi_n = 20
chain_model = "Discrete"
epsilon = 1.0   # a_A/a_B, conformational asymmetry

# Anderson Mixing
saddle_tolerance = 1e-7
saddle_max_iter  = 1
am_n_var = np.prod(nx) # W+
am_max_hist= 20
am_start_error = 1e-1
am_mix_min = 0.1
am_mix_init = 0.1

# -------------- initialize ------------
factory = PlatformSelector.create_factory("cuda")

# create instances
pc     = factory.create_polymer_chain(f, n_segment, chi_n, chain_model, epsilon)
cb     = factory.create_computation_box(nx, lx)
pseudo = factory.create_pseudo(cb, pc)
am     = factory.create_anderson_mixing(am_n_var,
            am_max_hist, am_start_error, am_mix_min, am_mix_init)

# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------")
print("Box Dimension: %d"  % (cb.get_dim()) )
print("chi_n: %f, f: %f, N: %d" % (pc.get_chi_n(), pc.get_f(), pc.get_n_segment()) )
print("Nx: %d, %d, %d" % (cb.get_nx(0), cb.get_nx(1), cb.get_nx(2)) )
print("Lx: %f, %f, %f" % (cb.get_lx(0), cb.get_lx(1), cb.get_lx(2)) )
print("dx: %f, %f, %f" % (cb.get_dx(0), cb.get_dx(1), cb.get_dx(2)) )
print("Volume: %f" % (cb.get_volume()) )

#-------------- allocate array ------------
# free end initial condition. q1 is q and q2 is qdagger.
# q1 starts from A end and q2 starts from B end.
q1_init = np.ones(cb.get_n_grid(), dtype=np.float64)
q2_init = np.ones(cb.get_n_grid(), dtype=np.float64)

# random seed for MT19937
np.random.seed(5489)

w_plus  = np.random.normal(0.0, 1.0, cb.get_n_grid())
w_minus = np.random.normal(0.0, 1.0, cb.get_n_grid())

# keep the level of field value
cb.zero_mean(w_plus)
cb.zero_mean(w_minus)

# find saddle point 
phi_a, phi_b, _, _, _, _, _ = DeepLangevinFTS.find_saddle_point(
    cb=cb, pc=pc, pseudo=pseudo, am=am,
    q1_init=q1_init, q2_init=q2_init, 
    w_plus=w_plus, w_minus=w_minus,
    max_iter=100,
    tolerance=saddle_tolerance,
    verbose_level=verbose_level)

# Tests for data_augmentation
print("Tests for data_augmentation")
w_plus = np.reshape(w_plus,   cb.get_nx()[-cb.get_dim():])
w_minus = np.reshape(w_minus, cb.get_nx()[-cb.get_dim():])
print(w_plus.shape)
print(w_minus.shape)

#------------------ Flip ----------------------
print("Flip: x direction")
X = np.flip(w_plus.copy(),  0)
Y = np.flip(w_minus.copy(), 0)
X = np.reshape(X, cb.get_n_grid())
Y = np.reshape(Y, cb.get_n_grid())
(phi_a, phi_b, _, _, _, _, _) \
    = DeepLangevinFTS.find_saddle_point(
        cb=cb, pc=pc, pseudo=pseudo, am=am, 
        q1_init=q1_init, q2_init=q2_init,
        w_plus=X, w_minus=Y,
        max_iter=saddle_max_iter,
        tolerance=saddle_tolerance,
        verbose_level=verbose_level)

print("Flip: y direction")
X = np.flip(w_plus.copy(),  1)
Y = np.flip(w_minus.copy(), 1)
X = np.reshape(X, cb.get_n_grid())
Y = np.reshape(Y, cb.get_n_grid())
(phi_a, phi_b, _, _, _, _, _) \
    = DeepLangevinFTS.find_saddle_point(
        cb=cb, pc=pc, pseudo=pseudo, am=am, 
        q1_init=q1_init, q2_init=q2_init,
        w_plus=X, w_minus=Y,
        max_iter=saddle_max_iter,
        tolerance=saddle_tolerance,
        verbose_level=verbose_level)

print("Flip: z direction")
X = np.flip(w_plus.copy(),  2)
Y = np.flip(w_minus.copy(), 2)
X = np.reshape(X, cb.get_n_grid())
Y = np.reshape(Y, cb.get_n_grid())
(phi_a, phi_b, _, _, _, _, _) \
    = DeepLangevinFTS.find_saddle_point(
        cb=cb, pc=pc, pseudo=pseudo, am=am, 
        q1_init=q1_init, q2_init=q2_init,
        w_plus=X, w_minus=Y,
        max_iter=saddle_max_iter,
        tolerance=saddle_tolerance,
        verbose_level=verbose_level)

#------------------ Transpose ----------------------
print("Transpose: x-y")
X = w_plus.copy().transpose(1,0,2)
Y = w_minus.copy().transpose(1,0,2)
X = np.reshape(X, cb.get_n_grid())
Y = np.reshape(Y, cb.get_n_grid())
(phi_a, phi_b, _, _, _, _, _) \
    = DeepLangevinFTS.find_saddle_point(
        cb=cb, pc=pc, pseudo=pseudo, am=am, 
        q1_init=q1_init, q2_init=q2_init,
        w_plus=X, w_minus=Y,
        max_iter=saddle_max_iter,
        tolerance=saddle_tolerance,
        verbose_level=verbose_level)

print("Transpose: x-z")
X = w_plus.copy().transpose(2,1,0)
Y = w_minus.copy().transpose(2,1,0)
X = np.reshape(X, cb.get_n_grid())
Y = np.reshape(Y, cb.get_n_grid())
(phi_a, phi_b, _, _, _, _, _) \
    = DeepLangevinFTS.find_saddle_point(
        cb=cb, pc=pc, pseudo=pseudo, am=am, 
        q1_init=q1_init, q2_init=q2_init,
        w_plus=X, w_minus=Y,
        max_iter=saddle_max_iter,
        tolerance=saddle_tolerance,
        verbose_level=verbose_level)

print("Transpose: y-z")
X = w_plus.copy().transpose(0,2,1)
Y = w_minus.copy().transpose(0,2,1)
X = np.reshape(X, cb.get_n_grid())
Y = np.reshape(Y, cb.get_n_grid())
(phi_a, phi_b, _, _, _, _, _) \
    = DeepLangevinFTS.find_saddle_point(
        cb=cb, pc=pc, pseudo=pseudo, am=am, 
        q1_init=q1_init, q2_init=q2_init,
        w_plus=X, w_minus=Y,
        max_iter=saddle_max_iter,
        tolerance=saddle_tolerance,
        verbose_level=verbose_level)

#------------------ Translation ----------------------
print("Translation: x-1")
X = np.roll(w_plus.copy(),  -1, axis=0)
Y = np.roll(w_minus.copy(), -1, axis=0)
X = np.reshape(X, cb.get_n_grid())
Y = np.reshape(Y, cb.get_n_grid())
(phi_a, phi_b, _, _, _, _, _) \
    = DeepLangevinFTS.find_saddle_point(
        cb=cb, pc=pc, pseudo=pseudo, am=am, 
        q1_init=q1_init, q2_init=q2_init,
        w_plus=X, w_minus=Y,
        max_iter=saddle_max_iter,
        tolerance=saddle_tolerance,
        verbose_level=verbose_level)

print("Translation: x+1")
X = np.roll(w_plus.copy(),  1, axis=0)
Y = np.roll(w_minus.copy(), 1, axis=0)
X = np.reshape(X, cb.get_n_grid())
Y = np.reshape(Y, cb.get_n_grid())
(phi_a, phi_b, _, _, _, _, _) \
    = DeepLangevinFTS.find_saddle_point(
        cb=cb, pc=pc, pseudo=pseudo, am=am, 
        q1_init=q1_init, q2_init=q2_init,
        w_plus=X, w_minus=Y,
        max_iter=saddle_max_iter,
        tolerance=saddle_tolerance,
        verbose_level=verbose_level)

print("Translation: y-1")
X = np.roll(w_plus.copy(),  -1, axis=1)
Y = np.roll(w_minus.copy(), -1, axis=1)
X = np.reshape(X, cb.get_n_grid())
Y = np.reshape(Y, cb.get_n_grid())
(phi_a, phi_b, _, _, _, _, _) \
    = DeepLangevinFTS.find_saddle_point(
        cb=cb, pc=pc, pseudo=pseudo, am=am, 
        q1_init=q1_init, q2_init=q2_init,
        w_plus=X, w_minus=Y,
        max_iter=saddle_max_iter,
        tolerance=saddle_tolerance,
        verbose_level=verbose_level)

print("Translation: y+1")
X = np.roll(w_plus.copy(),  1, axis=1)
Y = np.roll(w_minus.copy(), 1, axis=1)
X = np.reshape(X, cb.get_n_grid())
Y = np.reshape(Y, cb.get_n_grid())
(phi_a, phi_b, _, _, _, _, _) \
    = DeepLangevinFTS.find_saddle_point(
        cb=cb, pc=pc, pseudo=pseudo, am=am, 
        q1_init=q1_init, q2_init=q2_init,
        w_plus=X, w_minus=Y,
        max_iter=saddle_max_iter,
        tolerance=saddle_tolerance,
        verbose_level=verbose_level)

print("Translation: z-1")
X = np.roll(w_plus.copy(),  -1, axis=2)
Y = np.roll(w_minus.copy(), -1, axis=2)
X = np.reshape(X, cb.get_n_grid())
Y = np.reshape(Y, cb.get_n_grid())
(phi_a, phi_b, _, _, _, _, _) \
    = DeepLangevinFTS.find_saddle_point(
        cb=cb, pc=pc, pseudo=pseudo, am=am, 
        q1_init=q1_init, q2_init=q2_init,
        w_plus=X, w_minus=Y,
        max_iter=saddle_max_iter,
        tolerance=saddle_tolerance,
        verbose_level=verbose_level)

print("Translation: z+1")
X = np.roll(w_plus.copy(),  1, axis=2)
Y = np.roll(w_minus.copy(), 1, axis=2)
X = np.reshape(X, cb.get_n_grid())
Y = np.reshape(Y, cb.get_n_grid())
(phi_a, phi_b, _, _, _, _, _) \
    = DeepLangevinFTS.find_saddle_point(
        cb=cb, pc=pc, pseudo=pseudo, am=am, 
        q1_init=q1_init, q2_init=q2_init,
        w_plus=X, w_minus=Y,
        max_iter=saddle_max_iter,
        tolerance=saddle_tolerance,
        verbose_level=verbose_level)