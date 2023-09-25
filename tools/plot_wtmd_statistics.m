clear all;

load("wtmd_statistics_1000000.mat");

coeff_t = 1/dT+1;
exp_u = exp(u*coeff_t);
y = exp_u/sum(exp_u)/dpsi;

% Plot U(Psi)
figure(1);
plot(psi_range, u);
xlabel('\Psi')
ylabel('U(\Psi)')

% Plot U'(Psi)
figure(2);
plot(psi_range, up);
xlabel('\psi')
ylabel('U^\prime(\Psi)')

% Plot P(Psi)
figure(3);
plot(psi_range, y);
xlabel('\Psi')
ylabel('P(\Psi)')