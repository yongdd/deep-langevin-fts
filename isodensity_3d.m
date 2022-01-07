load("data/fields_010000.mat");

xsize = nx(1);
ysize = nx(2);
zsize = nx(3);

%Plot 3D
h=figure;
v = reshape(phi_a,[zsize, ysize, xsize]);
%v = permute(v,[1 2 3]);

[x,y,z] = meshgrid(1:ysize,1:zsize,1:xsize);
%isovalue = 0.0;
isovalue = 0.6;

x = double(x);
y = double(y);
z = double(z);

p1 = patch(isosurface(x,y,z,v,isovalue));
isonormals(x,y,z,v,p1)
set(p1,'FaceColor','b','EdgeColor','none');

p2 = patch(isocaps(x,y,z,v,isovalue),'FaceColor','interp',...
    'EdgeColor','none');

%Options
colormap jet
view(3);
axis off
lighting gouraud
axis([1 ysize 1 zsize 1 xsize])
set(gca,'XColor',[0 0 0],'XTick',[])
set(gca,'YColor',[0 0 0],'YTick',[])
set(gca,'ZColor',[0 0 0],'ZTick',[])

%Camera & Light
daspect([1 1 1])     % to change the ratio of axis
%daspect([width length height])
view(0,0)
%camlight left
%camlight(80,45)
view(20,50)
camlight right
%camlight left

set(h, 'PaperPositionMode', 'auto');
% [ auto | {manual} ]
set(h, 'PaperUnits', 'points');
% [ {inches} | centimeters | normalized | points ]
set(h, 'PaperPosition', [0 0 800 500]);
% [left,bottom,width,height]
% [~,outfilename,~] = fileparts(filename);
print (h,"isodensity",'-dpng') % print (h,'bulk','-dpdf')
% close(h)
