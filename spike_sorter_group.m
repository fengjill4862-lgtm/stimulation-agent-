% All rights reserved.
% Constraints:
% 1. Threshold: defined as a set value or SD*3 or SD*5
% 2. Artifact: defined as a set value (should not exceed)
% 3. Clear Region: duration and amplitude are both set
% 4. Post-spike amplitude: has to decay to a certain threshold post spike
% 5. Number of ripples: minimize post-spike oscillation

% Initialization
clear;
%close all;
set(0,'DefaultFigureWindowStyle','docked')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%FILE I/O%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File input
[fname pname]=uigetfile('*.*','Please select the Intan file to open');
cd(pname);
filenamelist=dir(pwd);
nel=length(filenamelist);
% File output
pnameToSave=uigetdir(pname,'Please select the folder to save the processed files');
numChannels=32;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

previousEndTime=0; % in second
allSpikes=cell(numChannels,1);
allIndices=cell(numChannels,1);

for i=3:nel
    currentFname=filenamelist(i).name
    if ~isdir(currentFname)
        results=spike_sorter_core(currentFname, pname,0,0);
        currentFname
        fnamePrefix=results.fnamePrefix;
        time=results.time;
        sampleRate=results.sampleRate;
        numChannels=results.numChannels;
        spikeTimeStamp=results.spikeTimeStamp;
        spikesGroup=results.spikesGroup;
        indicesGroup=results.indicesGroup;
        for traceIndex=1:numChannels
            allSpikes{traceIndex}=[allSpikes{traceIndex} results.spikesGroup{traceIndex}];
            allIndices{traceIndex}=[allIndices{traceIndex};
                indicesGroup{traceIndex}/sampleRate+previousEndTime];
        end
        previousEndTime=time(size(time,1),1)+previousEndTime;
    end
end


% save all channels
cd(pnameToSave);
for i=1:numChannels
    toSaveMat=[spikeTimeStamp allSpikes{i}];
    indexStr=num2str(i);
    while size(indexStr,2)<2
        indexStr=strcat('0',indexStr);
    end
    toSaveName=strcat('Spikes_Channel_',indexStr,'.dat');
    command=sprintf('save %s toSaveMat -ascii', toSaveName);
    eval(command);
    peakLocationName=strcat('Spike_Locations_Channel_',indexStr,'.txt');
    peakLocation=allIndices{i};
    command=sprintf('save %s peakLocation -ascii', peakLocationName);
    eval(command);
end
% remove noise at the same location
% artifact_loc_1=intersect(allIndices{1},allIndices{2});
% artifact_loc_2=intersect(allIndices{1},allIndices{23});