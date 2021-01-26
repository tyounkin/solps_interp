close all
clear all

gridr = ncread('interpolated_values.nc','gridr');
gridz = ncread('interpolated_values.nc','gridz');
values = ncread('interpolated_values.nc','values');

figure(1)
h = pcolor(gridr,gridz,values')
h.EdgeColor = 'none';
colorbar