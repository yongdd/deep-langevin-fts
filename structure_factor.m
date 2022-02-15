
load("data_simulation_chin17.59/structure_factor_010000.mat");
% Grid intervals should be the same
if (abs(lx(1)/nx(1) - lx(2)/nx(2)) + abs(lx(1)/nx(1) - lx(3)/nx(3)) + abs(lx(1)/nx(1) - lx(3)/nx(3)) > 1.e-5)
    print("Grid intervals should be the same")
end
v_mag = zeros(1, nx(1)^2 + nx(2)^2 + nx(3)^2);
v_mag_count = zeros(1, nx(1)^2 + nx(2)^2 + nx(3)^2);

for langevin_iter = 100000:10000:230000
    file_name = sprintf("data_simulation_chin17.59/structure_factor_%06d.mat", langevin_iter);
    disp(file_name)
    load(file_name);
    v = structure_factor;
    for i = 0:nx(1)-1
        for j = 0:nx(2)-1
            for k = 0:nx(3)/2
                v_mag(i^2+j^2+k^2+1) = v_mag(i^2+j^2+k^2+1) + v(i+1,j+1,k+1);
                v_mag_count(i^2+j^2+k^2+1) = v_mag_count(i^2+j^2+k^2+1) + 1;
            end
        end
    end
end

non_zero_points = v_mag_count > 0;
x2 = 0:nx(1)^2 + nx(2)^2 + nx(3)^2-1;
x = sqrt(double(x2(non_zero_points)))*2*pi/lx(1);
y2 = v_mag./v_mag_count;
y = y2(non_zero_points) + 0.0278;

h=figure;
semilogy(x,y);
%plot(x,y);
xlim([3 15])
ylim([0.01 1000])

%legend_list(end+1) = {sprintf('chi N = %d',chin)};
%hold on;