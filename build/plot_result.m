close all
clear all
compare = 1;
filename = 'interpolated_values.nc'
gridr = ncread(filename,'gridr');
gridz = ncread(filename,'gridz');
values = ncread(filename,'values');

figure(1)
h = pcolor(gridr,gridz,values')
h.EdgeColor = 'none';
colorbar

if(compare)
    filename = 'interpolated_values0.nc'
gridr = ncread(filename,'gridr');
gridz = ncread(filename,'gridz');
values2 = ncread(filename,'values');

figure(2)
h = pcolor(gridr,gridz,values2')
h.EdgeColor = 'none';
colorbar

figure(3)
h = pcolor(gridr,gridz,(values-values2)')
h.EdgeColor = 'none';
colorbar
end