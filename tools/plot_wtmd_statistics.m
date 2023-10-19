clear all;

load("wtmd_statistics_1000000.mat");

% Plot U(Ψ)
figure(1);
plot(psi_range, u);
xlabel('\Psi')
ylabel('U(\Psi)')

% Plot U'(Ψ)
figure(2);
plot(psi_range, up);
xlabel('\psi')
ylabel('U^\prime(\Psi)')

% Plot P(Ψ)
figure(3);

coeff_t = 1/dT+1;
exp_u = exp(u*coeff_t);
y = exp_u/sum(exp_u)/dpsi;

plot(psi_range, y);
xlabel('\Psi')
ylabel('P(\Psi)')

% Plot P(Ψ)
figure(4);

thrsheold = 1e-1;
I0 = I0/max(I0);
x = psi_range(I0 > threshold);
y = dH_psi_A_B(I0 > threshold);

plot(x, y);
xlabel('\Psi')
ylabel('\partial F/\partial{\it \chi_{b, AB} N}')