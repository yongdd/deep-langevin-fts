clear all;

% file_path = 'ABC_lamellar_-0.3/ABC_lammellar_chin_16.35_-0.3/data_simulation/structure_function';
file_path = 'data_simulation/w_diff';
file_name = strcat(file_path, sprintf('_%06d_%02d.mat', 1, 0));
load(file_name);

h=figure(2);
for s=0:5
    % Calculate k (Fourier mode) sqaure 
    k2         = zeros(nx(1),nx(2),floor(double(nx(3))/2)+1);
    k2_mapping = zeros(nx(1),nx(2),floor(double(nx(3))/2)+1);
    for i = 0:nx(1)-1
        for j = 0:nx(2)-1
            for k = 0:floor(double(nx(3))/2)
                k2(i+1,j+1,k+1) = round((double(i)/lx(1))^2 + (double(j)/lx(2))^2 + (double(k)/lx(3))^2,7);
            end
        end
    end

    % Remove duplicates and set mapping
    k2_unique = unique(k2);
    for i = 0:nx(1)-1
        for j = 0:nx(2)-1
            for k = 0:floor(double(nx(3))/2)
                idx = find(k2_unique == round((double(i)/lx(1))^2 + (double(j)/lx(2))^2 + (double(k)/lx(3))^2,7));
                k2_mapping(i+1,j+1,k+1) = idx;
            end
        end
    end

    % Read data and caculate averages
    sf_mag   = zeros(size(k2_unique));
    sf_count = zeros(size(k2_unique));
    for n = 1:1:500
        files = dir(strcat(file_path, sprintf('_%06d_*.mat', n)));
        
        if length(files) ~= 12
            continue
        end
        
        if s == 0
            file_name = fullfile(files(1).folder, files(1).name);
        elseif s == 1
            file_name = fullfile(files(1).folder, files(2).name);
        elseif s == 2
            file_name = fullfile(files(1).folder, files(3).name);
        elseif s == 3
            file_name = fullfile(files(1).folder, files(4).name);
        elseif s == 4
            file_name = fullfile(files(1).folder, files(5).name);
%         elseif s == 3
%             file_name = fullfile(files(1).folder, files((round(length(files)/3*1))).name);
%         elseif s == 4
%             file_name = fullfile(files(1).folder, files((round(length(files)/3*2))).name);
        elseif s == 5
            file_name = fullfile(files(1).folder, files(end).name);
        else
            continue;
        end
        
        disp(file_name)
        load(file_name);
        % w_diff = reshape(w_diff,nx);
        % w_diff_k = fftn(w_diff);

        v = w_diff_mag2;
%         v = real(w_diff_mag2/std(w_diff)^2);

        for i = 0:nx(1)-1
            for j = 0:nx(2)-1
                for k = 0:floor(double(nx(3))/2)
                    idx = k2_mapping(i+1,j+1,k+1);
                    sf_mag(idx) = sf_mag(idx) + v(i+1,j+1,k+1);
                    sf_count(idx) = sf_count(idx) + 1;
                end
            end
        end

    end

    x = sqrt(double(k2_unique))*2*pi;
%     x = 2*pi/x;
    y = sf_mag./sf_count;
    % y(1) = 0.0;
    % 
    %semilogy(x(2:end),y(2:end));
    loglog(x(1:end),y(1:end))
    hold on;
end
xlim([0.8 20])
ylim([1e-15 0.1])