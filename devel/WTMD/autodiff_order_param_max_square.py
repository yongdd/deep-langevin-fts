import time
import torch
import numpy as np

nx = [31,36,39]
lx = [5,4,3]

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

time_start = time.time()

for i in range(100):
    w_minus = torch.rand(nx, requires_grad=True)
    w_minus = w_minus - torch.mean(w_minus)
    w_minus_k = torch.fft.rfftn(w_minus)/np.prod(nx)

    w_minus_squared = (w_minus_k*torch.conj(w_minus_k)).real
    w_minus_max = torch.max(w_minus_squared)
    w_minus_argmax = torch.argmax(w_minus_squared)

    kx_star = space_kx[w_minus_argmax]
    ky_star = space_ky[w_minus_argmax]
    kz_star = space_kz[w_minus_argmax]

    manual_grad = 1.0/np.prod(nx)*w_minus_k.detach().cpu().numpy().flatten()[w_minus_argmax]*np.exp(1.0j*(space_x*kx_star+space_y*ky_star+space_z*kz_star))
    manual_grad = 2*manual_grad.real
    auto_grad = torch.autograd.grad(w_minus_max, w_minus)
    
    # w_minus = np.random.uniform(-1, 1, nx)
    # w_minus = w_minus - np.mean(w_minus)
    # w_minus_k = np.fft.rfftn(w_minus)/np.prod(nx)

    # w_minus_squared = (w_minus_k*np.conj(w_minus_k)).real
    # w_minus_max = np.max(w_minus_squared)
    # w_minus_argmax = np.argmax(w_minus_squared)

    # kx_star = space_kx[w_minus_argmax]
    # ky_star = space_ky[w_minus_argmax]
    # kz_star = space_kz[w_minus_argmax]

    # manual_grad = 1.0/np.prod(nx) * w_minus_k.flatten()[w_minus_argmax]*np.exp(1.0j*(space_x*kx_star+space_y*ky_star+space_z*kz_star))
    # manual_grad = 2*manual_grad.real

    # print(w_minus)
    # print(w_minus_k)
    # print(w_minus_k_real)
    # print(w_minus_squared)
    # print(w_minus_argmax, torch.flatten(w_minus_squared)[w_minus_argmax], w_minus_max)

    # print(w_minus.shape)
    # print(w_minus_k.shape)
    # print(w_minus_squared.shape)

    # print(space_y.shape, space_x.shape, space_z.shape)
    # print(space_ky.shape, space_kx.shape, space_kz.shape)

    # print(autograd[0].shape)
    # print(manual_grad.shape)

    # print(autograd[0])
    # print(manual_grad)
    print(i, torch.std(auto_grad[0]-manual_grad).item())
    # print(i)
    
time_elapsed = time.time() - time_start
print(time_elapsed)