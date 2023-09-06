import time
import torch
import numpy as np

nx = [40,40,40]
# nx = [2,3,4]
lx = [4.38,4.38,4.38]

# Set initial fields
with open('input') as f:
    lines = f.readlines()
    fields = np.array([float(x) for x in lines])
    input_fields = np.reshape(np.array([float(x) for x in lines]), ([2]+nx))

# w_A = input_fields[1,:,:,:] + input_fields[0,:,:,:]
# w_B = input_fields[1,:,:,:] - input_fields[0,:,:,:]

ell=4
kc=6.02

m = nx
L = lx
M = m[0]*m[1]*m[2]
Mk = m[0]*m[1]*(m[2]//2+1)

K = np.zeros(Mk)
wt = np.zeros(Mk)
fk = np.zeros(Mk)

wt[:] = 2.0

for k0 in range(-(m[0]-1)//2, m[0]//2+1):
    if k0<0:
        K0 = k0+m[0]
    else:
        K0 = k0
    kx_sq = k0*k0/(L[0]*L[0])
    for k1 in range(-(m[1]-1)//2, m[1]//2+1):
        if k1<0:
            K1 = k1+m[1]
        else:
            K1 = k1
        ky_sq = k1*k1/(L[1]*L[1])
        for k2 in range(0, m[2]//2+1):
            kz_sq = k2*k2/(L[2]*L[2])
            k = k2+(m[2]//2+1)*(K1+m[1]*K0)
            K[k] = 2*np.pi*np.power(kx_sq+ky_sq+kz_sq, 0.5)
            if k2==0 or k2==m[2]//2:
                wt[k]=1
            fk[k] = 1.0/(1.0 + np.exp(12.0*(K[k]-kc)/kc))

# print(np.mean(K), np.std(K))
# print(np.mean(wt), np.std(wt))
# print(np.mean(fk), np.std(fk))

wt = torch.tensor(np.reshape(wt, (m[0],m[1],m[2]//2+1)))
fk = torch.tensor(np.reshape(fk, (m[0],m[1],m[2]//2+1)))

V = np.prod(lx)
M = np.prod(nx)

time_start = time.time()

for i in range(1):

    # w_minus = torch.rand(nx, requires_grad=True)
    # w_minus = w_minus - torch.mean(w_minus)

    w_minus = torch.tensor(input_fields[0,:,:,:], requires_grad=True)*2
    w_minus_k = torch.fft.rfftn(w_minus)
    
    Psi = torch.sum(torch.pow(torch.absolute(w_minus_k), ell)*fk*wt/M)
    Psi = torch.pow(Psi/M, 1.0/ell)

    # print(w_minus_k[0,0,10].item(), ell, fk[0,0,10].item(), wt[0,0,10].item())

    auto_grad = torch.autograd.grad(Psi*M/V, w_minus)
    
    w_minus_k_np = w_minus_k.detach().cpu().numpy()
    fk_np = fk.detach().cpu().numpy()
    Psi_np = Psi.detach().cpu().numpy()
    
    dPsi_dwk = np.power(np.absolute(w_minus_k_np), ell-2.0) * np.power(Psi_np,1.0-ell)*w_minus_k_np*fk_np
    dPsi_dwr = np.fft.irfftn(dPsi_dwk, m)/V
    manual_func_deriv = dPsi_dwr

    print(Psi.item(), manual_func_deriv[0,0,10])

    # w_minus = np.random.uniform(-1, 1, nx)
    # w_minus = w_minus - np.mean(w_minus)
    # w_minus_k = np.fft.rfftn(w_minus)/np.prod(nx)

    # w_minus_squared = (w_minus_k*np.conj(w_minus_k)).real
    # w_minus_max = np.max(w_minus_squared)
    # w_minus_argmax = np.argmax(w_minus_squared)

    # kx_star = space_kx[w_minus_argmax]
    # ky_star = space_ky[w_minus_argmax]
    # kz_star = space_kz[w_minus_argmax]

    # manual_func_deriv = 1.0/np.prod(nx) * w_minus_k.flatten()[w_minus_argmax]*np.exp(1.0j*(space_x*kx_star+space_y*ky_star+space_z*kz_star))
    # manual_func_deriv = 2*manual_func_deriv.real

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
    # print(manual_func_deriv.shape)

    # print(auto_grad[0])
    # print(manual_func_deriv)
    print(manual_func_deriv[0,0,10], auto_grad[0][0,0,10])
    
    print(i, torch.std(auto_grad[0]-manual_func_deriv).item(), torch.std(auto_grad[0]), np.std(manual_func_deriv))
    # print(i)
    
time_elapsed = time.time() - time_start
print(time_elapsed)