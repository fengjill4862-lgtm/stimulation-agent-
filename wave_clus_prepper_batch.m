% Prepares the 'ASCII spikes' file in '.mat' format for wave_clus analysis.
% Basic work flow:
% 1) It takes the spikes file for a given channel and store the spikes in
% Nx61 matrix as the .spikes field of the toSaveMat structure.
% 2) It takes the spike locations file for the same channel and store all
% spike locations in a 1xN vector as the .index field of the toSaveMat
% structure.
% 3) Then it saves the toSaveMat as a structure in .mat format that can be
% opened by wave_clus for clustering. 

clear
close all
[fname pnameSpikes]=uigetfile('*.*','Get the spikes file');
cd(pnameSpikes)
filenamelistSpikes=dir(pwd);
nelSpikes=length(filenamelistSpikes);

[fname pnameLocations]=uigetfile('*.*','Get the spike locations file');
cd(pnameLocations)
filenamelistLocations=dir(pwd);
nelLocations=length(filenamelistLocations);

pnameToSave=uigetdir(pnameSpikes,'Select the path to save');

if nelSpikes==nelLocations
    nel=nelSpikes;
    for i=3:nel
        cd(pnameSpikes)
        fname=filenamelistSpikes(i).name;
        tempMat=load(fname)';
        toSaveMat.spikes=tempMat(2:size(tempMat,1),:);
        cd(pnameLocations)
        fname=filenamelistLocations(i).name;
        toSaveMat.index=load(fname)'*1000;
        cd(pnameToSave)
        indexStr=num2str(i-2);
        while size(indexStr,2)<2
            indexStr=strcat('0',indexStr);
        end
        toSaveName=strcat('Ch',indexStr,'_spikes_for_wave_clus.mat');
        save(toSaveName,'-struct','toSaveMat');
    end
end