import time
import torch
import numpy as np
from scipy.io import savemat, loadmat

# nx = [31,36,39]
nx = [3,4,5]
lx = [1.2,1.4,1.3]

# V = np.prod(lx)
# M = np.prod(nx)
# f_x = torch.rand(nx, requires_grad=True)
# func_int =  V/M*torch.sum(f_x)
# auto_grad = torch.autograd.grad(func_int, f_x)
# print(auto_grad[0]*M/V)

nx = [40,40,40]
lx = [4.38,4.38,4.38]

# Set initial fields
with open('input') as f:
    lines = f.readlines()
    fields = np.array([float(x) for x in lines])
    input_fields = np.reshape(np.array([float(x) for x in lines]), ([2]+nx))

# nx = [40,40,40]
# lx = [4.36,4.36,4.36]

# input_data = loadmat("lamella_equil_chin17.0.mat", squeeze_me=True)
# w_A = input_data["w_plus"] + input_data["w_minus"]

space_x, space_y, space_z = np.meshgrid(
    lx[0]*np.arange(nx[0])/nx[0],
    lx[1]*np.arange(nx[1])/nx[1],
    lx[2]*np.arange(nx[2])/nx[2], indexing='ij')

space_kx, space_ky, space_kz = np.meshgrid(
    2*np.pi/lx[0]*np.arange(nx[0]),
    2*np.pi/lx[1]*np.arange(nx[1]),
    2*np.pi/lx[2]*np.arange(nx[2]//2+1), indexing='ij')

space_kx = space_kx.flatten()
space_ky = space_ky.flatten()
space_kz = space_kz.flatten()

V = np.prod(lx)
M = np.prod(nx)

time_start = time.time()

for i in range(1):
    # w_minus = torch.rand(nx, requires_grad=True)*10
    # w_minus = w_minus - torch.mean(w_minus)

    w_minus = torch.tensor(input_fields[0,:,:,:], requires_grad=True)*2
    # w_minus = torch.reshape(torch.tensor(input_data["w_minus"], requires_grad=True), nx)
    w_minus_k = torch.fft.rfftn(w_minus)
    
    w_minus_squared_M = torch.sqrt((w_minus_k*torch.conj(w_minus_k)).real)/M
    w_minus_max_M = torch.max(w_minus_squared_M)
    w_minus_argmax = torch.argmax(w_minus_squared_M)

    kx_star = space_kx[w_minus_argmax]
    ky_star = space_ky[w_minus_argmax]
    kz_star = space_kz[w_minus_argmax]

    manual_func_deriv = 0.5/M/V* \
        w_minus_k.detach().cpu().numpy().flatten()[w_minus_argmax]/ \
        w_minus_max_M.detach().cpu().numpy()* \
        np.exp(1.0j*(space_x*kx_star+space_y*ky_star+space_z*kz_star))
    manual_func_deriv = 2*manual_func_deriv.real
    auto_grad = torch.autograd.grad(w_minus_max_M*M/V, w_minus)
    
    print(w_minus_k.detach().cpu().numpy().flatten()[w_minus_argmax], w_minus_max_M.detach().cpu().numpy() )
    
    # manual_func_deriv = 0.5/np.prod(nx)* \
    #     w_minus_k.detach().cpu().numpy().flatten()[w_minus_argmax]/ \
    #     w_minus_max_M.detach().cpu().numpy()* \
    #     np.exp(1.0j*(space_x*kx_star+space_y*ky_star+space_z*kz_star))
    # manual_func_deriv = 2*manual_func_deriv.real
    
    # auto_grad = torch.autograd.grad(w_minus_max_M, w_minus_k)
    
    print(w_minus_max_M.item(), manual_func_deriv[0,0,10].item())
    
    # w_minus = np.random.uniform(0, 1, nx)
    # w_minus = w_minus - np.mean(w_minus)
    # w_minus_k = np.fft.rfftn(w_minus)

    # w_minus_squared_M = np.sqrt((w_minus_k*np.conj(w_minus_k)).real/np.prod(nx))
    # w_minus_max_M = np.max(w_minus_squared_M)
    # w_minus_argmax = np.argmax(w_minus_squared_M)

    # kx_star = space_kx[w_minus_argmax]
    # ky_star = space_ky[w_minus_argmax]
    # kz_star = space_kz[w_minus_argmax]

    # manual_func_deriv = 0.5/w_minus_max_M/np.prod(nx)*w_minus_k.flatten()[w_minus_argmax]*np.exp(1.0j*(space_x*kx_star+space_y*ky_star+space_z*kz_star))
    # manual_func_deriv = 2*manual_func_deriv.real

    # print(w_minus)
    # print(w_minus_k)
    # print(w_minus_squared_M)
    # print(w_minus_argmax, torch.flatten(w_minus_squared_M)[w_minus_argmax], w_minus_max_M)

    # print(w_minus.shape)
    # print(w_minus_k.shape)
    # print(w_minus_squared_M.shape)

    # print(space_y.shape, space_x.shape, space_z.shape)
    # print(space_ky.shape, space_kx.shape, space_kz.shape)

    # print(autograd[0].shape)
    # print(manual_func_deriv.shape)

    # print(auto_grad[0])
    # print(manual_func_deriv)
    # print(*M/V*auto_grad[0]/manual_func_deriv)
    print(auto_grad[0][0,0,10], manual_func_deriv[0,0,10])
    
    print(i, torch.std(auto_grad[0]-manual_func_deriv).item(), torch.std(auto_grad[0]), np.std(manual_func_deriv))
    # print(i, np.std(manual_func_deriv))
    
time_elapsed = time.time() - time_start
print(time_elapsed)