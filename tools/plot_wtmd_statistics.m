clear all;

load("wtmd_statistics_1000000.mat");

coeff_t = 1/DT+1;
exp_u = exp(u*coeff_t);
y = exp_u/sum(exp_u)/dPsi;

% Plot U(Psi)
figure(1);
plot(Psi_range, u);
xlabel('\Psi')
ylabel('U(\Psi)')

% Plot U'(Psi)
figure(2);
plot(Psi_range, up);
xlabel('\Psi')
ylabel('U^\prime(\Psi)')

% Plot P(Psi)
figure(3);
plot(Psi_range, y);
xlabel('\Psi')
ylabel('P(\Psi)')