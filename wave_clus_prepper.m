% Written by Guosong on July 7, 2015

% Prepares the 'ASCII spikes' file in '.mat' format for wave_clus analysis.
% Basic work flow:
% 1) It takes the spikes file for a given channel and store the spikes in
% Nx61 matrix as the .spikes field of the toSaveMat structure.
% 2) It takes the spike locations file for the same channel and store all
% spike locations in a 1xN vector as the .index field of the toSaveMat
% structure.
% 3) Then it saves the toSaveMat as a structure in .mat format that can be
% opened by wave_clus for clustering. 

% Written by Guosong Hong on July 7, 2015
clear
close all
[fname pname]=uigetfile('*.*','Get the spikes file');
cd(pname)
tempMat=load(fname)';
toSaveMat.spikes=tempMat(2:size(tempMat,1),:);

[fname pname]=uigetfile('*.*','Get the spike locations file');
cd(pname)
toSaveMat.index=load(fname)'*1000;

pnameToSave=uigetdir(pname,'Select the path to save');
cd(pnameToSave)
save('Ch15_spikes_for_wave_clus.mat','-struct','toSaveMat');