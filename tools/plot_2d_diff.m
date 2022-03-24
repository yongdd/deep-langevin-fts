%Load File
load("diff_dl.mat");
%load("diff_am.mat");

%Plot 3D
h=figure;
data = wpd;
disp([max(max(max(data)))])
disp([min(min(min(data)))])
v = reshape(data,[nx(3), nx(2), nx(1)]);

a_concentration = reshape(v(:,:,1),[nx(2), nx(3)]);
image(a_concentration(:,:),'CDataMapping','scaled');
%shading flat;
%shading interp;
set(gca,'DataAspectRatio',[1 1 1]);
axis off;
colormap jet;
colorbar('FontSize',20)
%caxis([-1.8 1.5])
caxis([-0.45 0.45])

set(h, 'PaperPositionMode', 'auto');
% [ auto | {manual} ]
set(h, 'PaperUnits', 'points');
% [ {inches} | centimeters | normalized | points ]
set(h, 'PaperPosition', [0 0 500 500]);
% [left,bottom,width,height]
%[~,outfilename,~] = fileparts(filename);
print (h,"diff_gt",'-dpng') % print (h,'bulk','-dpdf')
%print (h,strcat("pressure_", outfilename),'-dpng') % print (h,'bulk','-dpdf')
%close(h)