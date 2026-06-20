% Turned into a 2-arg function on July 23, 2015.
% All rights reserved.
% Constraints:
% 1. Threshold: defined as a set value or median/0.6745*4 or a fixed value
% 2. Artifact: defined as a set value (should not exceed)
% 3. Clear Region: duration and amplitude are both set
% 4. Post-spike amplitude: has to decay to a certain threshold post spike
% 5. Number of ripples: minimize post-spike oscillation

function returnPackage=spike_sorter_core(fname,pname,startTime,endTime);

%%%%%%%%%%%%%%%%%%%%%%%%%PARAMETER INITIALIZATIONS%%%%%%%%%%%%%%%%%%%%%%%%%
% Change the start and end times for spike sorting
% Values input here are in seconds
startSortingTime=5e-5+startTime; % Note: startSortingTime has to be non-zero. So if one wants to start from 0, put 5e-5 instead.
endSortingTime=endTime;

% Change what channels to inspect
channelOption='all'; % Available inputs: 'all' or 'selected'
% Channel indices. Only used in 'selected' mode. If rearrange channels,
% need channel# after arrangement here
channels=[13:14];
%channels=[4:29]; %1/1 right peptide or 3/28 left CD11b after rearrangement take 26 channels in middle (total 32)
%channels=[1:2,4:7,9:14,17:29,31]; %3/31 left unmod after rearrangement use all 26 channels
%channels=[1:11,13:19,21,23:25,28:31];% 01/05,right EAAT2 after rearrangement take 26 channels (total 27)
%channels=[2:6,8:18,20:22,24:27,29:31];%3/30 right control Ab after rearrangement take 26 channels (total 27)
%channels=[4,8:11,13:14,17:20,22,24:28,31]; %20200213_2_v9 Left D2 ETIC after rearrangement 18 channels
%channels=[1:14,16:18,21:32]; %20191213_2_v9 Left unmod ETIC after rearrangement 29 channels
%channels=[1:14,16:26,31:32]; 
%channels=[6:12,15:20,22:27,29:31]; %12/23 left
%channels=[21,29];
%channels=[15:28,30];%0914_1,D2
%channels=[4:6,8:9,11:13,15:21,23,26:32];%0913_1,D2
%channels=[2:6,10:17,20:26,31];%10/31_1,D2
%channels=[2:7,9:13,16:18,20,23,25:30];%10/31_2,D2
%channels=[1:19,21,24,29:31];%1103_1,unmod
%channels=[3:9,20:23,25:27,29:32];%1101_2, unmod, v11
%channels=[1:3,5,8:10,12:14,16:20,25:26,28:32];%1102_1_v11_D2
%channels=[2,4:8,10:26];
%channels=[1:24,29:31];
%channels=[1:16,18:19,21:22,25:27,29:31];%03/31,unmod
%channels=[1:9,11:24,29:30,32];%0330,control Ab,26channels(extra:Ch31)
%channels=[1:3,8:13,15:31];%0105,EAAT2,26channels(extra:Ch32)
%channels=[1];
%channels=[1:14,16:19,21:23];%1126_1,stroke recording
%channels=[1:6,8:13,18,21:23];%1126_1,stroke recording, 16 Ch for plotting
%channels=[3:15,18:24,27:29];%1125_1,stroke recording
%channels=[1:6,8];%1125_3,stroke recording
%channels=[2:3,5,9:18,20:22,25,27:32];%1206_v21_D2
%channels=[3:18,22:32];%1204_2,stroke recording
%channels=[3:6,10,11,14,15,18,22:26,29,30];%1204_2,stroke recording
%channels=[1:2,4:16,19:28,30:32,33:34,36,39,41:42,44,46,48,52,55:62,64];%1213_2,unmod,v9
%channels=[2,3,5,9:16,18,22,24,27,29:31];%0213_2_D2
%channels=[1:16,19:28,30:32];%1213_2,unmod,left


% Time and amplitude constraints places on spike sorting
preSpike=20; % pre-spike time duration is 1 ms (20/20000)
postSpike=40; % post-spike time duration is 2 ms (40/20000)
clearRegion=1; % 1.0 ms interval required between two neighboring spikes
detectMode='both'; % Detect positive only, negative only or both signs of peaks
thresholdMode='Median'; % Available inputs: 'SetValue' or 'Median' 
thresholdSetValue = 10; % in uV. Only used in 'SetValue' mode.
artifactCutoff = 1500; % Maximum amplitude for artifact removal, in uV.

% Bandpass filter initialization
% sampleRate is read from the Intan file header after loading the data.
passBand=[250 6000]; % Hzjetblue.
filterOrder=1; 
filterIter=11;
ftype='bandpass';
filterName='butterworth'; % butterworth, elliptic or chebyshev
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

packagePath=fileparts(mfilename('fullpath'));
if ~isempty(packagePath)
    addpath(packagePath); % Keep sibling Intan readers available after cd(pname).
end
cd(pname);
[~, fnamePrefix, fileExtension]=fileparts(fname);
switch lower(fileExtension)
    case '.rhd'
        [originalData, frequency_parameters]=read_Intan_RHD2000_file_no_prompt_new(fname,pname); % rows: channels; columns: times
    case '.rhs'
        [originalData, frequency_parameters]=read_Intan_RHS2000_file_no_prompt_new(fname,pname); % rows: channels; columns: times
    otherwise
        error('Unsupported Intan file extension "%s". Expected .rhd or .rhs.', fileExtension);
end
sampleRate=frequency_parameters.amplifier_sample_rate; % Hz

% read_Intan_RHD2000_file_new;
% originalData=amplifier_data;
% adcData=read_Intan_RHD2000_file_adc(fname,pname);
size(originalData)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CHANNEL REORDERING %%%%%%%%%%%%%%%%%%%%%%%%%%
%%rearrange for cell targeting paper, meshes on the RIGHT
% dataBuffer=originalData;
% originalData=[];
% originalData(1,:)=dataBuffer(28,:);
% originalData(2,:)=dataBuffer(11,:);
% originalData(3,:)=dataBuffer(20,:);
% originalData(4,:)=dataBuffer(1,:);
% originalData(5,:)=dataBuffer(3,:);
% originalData(6,:)=dataBuffer(30,:);
% originalData(7,:)=dataBuffer(27,:);
% originalData(8,:)=dataBuffer(12,:);
% originalData(9,:)=dataBuffer(2,:);
% originalData(10,:)=dataBuffer(19,:);
% originalData(11,:)=dataBuffer(9,:);
% originalData(12,:)=dataBuffer(4,:);
% originalData(13,:)=dataBuffer(18,:);
% originalData(14,:)=dataBuffer(29,:);
% originalData(15,:)=dataBuffer(22,:);
% originalData(16,:)=dataBuffer(13,:);
% originalData(17,:)=dataBuffer(32,:);
% originalData(18,:)=dataBuffer(17,:);
% originalData(19,:)=dataBuffer(10,:);
% originalData(20,:)=dataBuffer(5,:);
% originalData(21,:)=dataBuffer(21,:);
% originalData(22,:)=dataBuffer(14,:);
% originalData(23,:)=dataBuffer(31,:);
% originalData(24,:)=dataBuffer(15,:);
% originalData(25,:)=dataBuffer(24,:);
% originalData(26,:)=dataBuffer(6,:);
% originalData(27,:)=dataBuffer(7,:);
% originalData(28,:)=dataBuffer(26,:);
% originalData(29,:)=dataBuffer(16,:);
% originalData(30,:)=dataBuffer(23,:);
% originalData(31,:)=dataBuffer(8,:);
% originalData(32,:)=dataBuffer(25,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CHANNEL REORDERING %%%%%%%%%%%%%%%%%%%%%%%%%%
%%rearrange for cell targeting paper, meshes on the LEFT
% dataBuffer=originalData;
% originalData=[];
% originalData(32,:)=dataBuffer(28,:);
% originalData(31,:)=dataBuffer(11,:);
% originalData(30,:)=dataBuffer(20,:);
% originalData(29,:)=dataBuffer(1,:);
% originalData(28,:)=dataBuffer(3,:);
% originalData(27,:)=dataBuffer(30,:);
% originalData(26,:)=dataBuffer(27,:);
% originalData(25,:)=dataBuffer(12,:);
% originalData(24,:)=dataBuffer(2,:);
% originalData(23,:)=dataBuffer(19,:);
% originalData(22,:)=dataBuffer(9,:);
% originalData(21,:)=dataBuffer(4,:);
% originalData(20,:)=dataBuffer(18,:);
% originalData(19,:)=dataBuffer(29,:);
% originalData(18,:)=dataBuffer(22,:);
% originalData(17,:)=dataBuffer(13,:);
% originalData(16,:)=dataBuffer(32,:);
% originalData(15,:)=dataBuffer(17,:);
% originalData(14,:)=dataBuffer(10,:);
% originalData(13,:)=dataBuffer(5,:);
% originalData(12,:)=dataBuffer(21,:);
% originalData(11,:)=dataBuffer(14,:);
% originalData(10,:)=dataBuffer(31,:);
% originalData(9,:)=dataBuffer(15,:);
% originalData(8,:)=dataBuffer(24,:);
% originalData(7,:)=dataBuffer(6,:);
% originalData(6,:)=dataBuffer(7,:);
% originalData(5,:)=dataBuffer(26,:);
% originalData(4,:)=dataBuffer(16,:);
% originalData(3,:)=dataBuffer(23,:);
% originalData(2,:)=dataBuffer(8,:);
% originalData(1,:)=dataBuffer(25,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CHANNEL REORDERING %%%%%%%%%%%%%%%%%%%%%%%%%%
%%rearrange for acute1__190520_124219 
% dataBuffer=originalData;
% originalData=[];
% originalData(1,:)=dataBuffer(2,:);
% originalData(2,:)=dataBuffer(15,:);
% originalData(3,:)=dataBuffer(16,:);
% originalData(4,:)=dataBuffer(5,:);
% originalData(5,:)=dataBuffer(13,:);
% originalData(6,:)=dataBuffer(14,:);
% originalData(7,:)=dataBuffer(7,:);
% originalData(8,:)=dataBuffer(1,:);
% originalData(9,:)=dataBuffer(6,:);
% originalData(10,:)=dataBuffer(12,:);
% originalData(11,:)=dataBuffer(4,:);
% originalData(12,:)=dataBuffer(8,:);
% originalData(13,:)=dataBuffer(3,:);
% originalData(14,:)=dataBuffer(10,:);
% originalData(15,:)=dataBuffer(11,:);
% originalData(16,:)=dataBuffer(9,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CHANNEL REORDERING %%%%%%%%%%%%%%%%%%%%%%%%%%
%%rearrange for acute1_190702_121611 
% dataBuffer=originalData;
% originalData=[];
% originalData(1,:)=dataBuffer(1,:);
% originalData(2,:)=dataBuffer(16,:);
% originalData(3,:)=dataBuffer(2,:);
% originalData(4,:)=dataBuffer(15,:);
% originalData(5,:)=dataBuffer(3,:);
% originalData(6,:)=dataBuffer(14,:);
% originalData(7,:)=dataBuffer(4,:);
% originalData(8,:)=dataBuffer(13,:);
% originalData(9,:)=dataBuffer(5,:);
% originalData(10,:)=dataBuffer(12,:);
% originalData(11,:)=dataBuffer(6,:);
% originalData(12,:)=dataBuffer(11,:);
% originalData(13,:)=dataBuffer(7,:);
% originalData(14,:)=dataBuffer(10,:);
% originalData(15,:)=dataBuffer(8,:);
% originalData(16,:)=dataBuffer(9,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CHANNEL REORDERING %%%%%%%%%%%%%%%%%%%%%%%%%%
%rearrange for Rat 92 (after new connecter)
% dataBuffer=originalData;
% originalData=[];
% originalData(1,:)=dataBuffer(16,:);
% originalData(2,:)=dataBuffer(1,:);
% originalData(3,:)=dataBuffer(15,:);
% originalData(4,:)=dataBuffer(2,:);
% originalData(5,:)=dataBuffer(14,:);
% originalData(6,:)=dataBuffer(3,:);
% originalData(7,:)=dataBuffer(13,:);
% originalData(8,:)=dataBuffer(4,:);
% originalData(9,:)=dataBuffer(12,:);
% originalData(10,:)=dataBuffer(5,:);
% originalData(11,:)=dataBuffer(11,:);
% originalData(12,:)=dataBuffer(6,:);
% originalData(13,:)=dataBuffer(10,:);
% originalData(14,:)=dataBuffer(7,:);
% originalData(15,:)=dataBuffer(9,:);
% originalData(16,:)=dataBuffer(8,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if startTime==endTime
    startSortingTime=1/sampleRate;
    endSortingTime=size(originalData,2)/sampleRate;
end
switch channelOption
    case 'selected'
        transposedData=originalData(channels,round(startSortingTime*sampleRate):round(endSortingTime*sampleRate))'; % rows: times; columns: channels
    case 'all'
        transposedData=originalData(:,round(startSortingTime*sampleRate):round(endSortingTime*sampleRate))'; % rows: times; columns: channels
    % Note: 'round' is used here to avoid precision carry error.
end
[numPoints numChannels]=size(transposedData);
t=(startSortingTime:1/sampleRate:endSortingTime)';

% Filter the original traces and save both unfiltered and filtered traces
filteredData=transposedData;
size(filteredData)
Wn=2*passBand./sampleRate;  % Normalized cutoff frequency. Unitless. 
[butterB,butterA]=butter(filterOrder,Wn,ftype);    %prepare the butterworth filter.
[ellipB,ellipA]=ellip(filterOrder,0.1,100,Wn,ftype);  %prepare the elliptical filter (2nd par: ripple amp; 3rd par: stopband atten).
[chebyB,chebyA]=cheby2(filterOrder,18,Wn,ftype);  %prepare the type-2 Chebyshev filter. (2nd par: stopband atten).
for i=1:filterIter
    switch filterName
        case 'butterworth'
            filteredData=filtfilt(butterB,butterA,filteredData);
        case 'elliptic'
            filteredData=filtfilt(ellipB,ellipA,filteredData);
        case 'chebyshev'
            filteredData=filtfilt(chebyB,chebyA,filteredData);
    end        
end

filteredDataBeforeCAR=filteredData;

% Apply CAR after filter
% cars=zeros(numPoints,numChannels);
% cars(:,1)=mean(filteredData(:,[3:6])')';
% cars(:,2)=mean(filteredData(:,[4:7])')';
% cars(:,3)=mean(filteredData(:,[1 5 6 7])')';
% cars(:,numChannels-2)=mean(filteredData(:,[numChannels-6 numChannels-5 numChannels-4 numChannels])')';
% cars(:,numChannels-1)=mean(filteredData(:,[numChannels-6:numChannels-3])')';
% cars(:,numChannels)=mean(filteredData(:,[numChannels-5:numChannels-2])')';
% for i=4:numChannels-3
%     cars(:,i)=mean(filteredData(:,[i-3 i-2 i+2 i+3])')';
% end
% filteredData=filteredData-cars;
% size(filteredData)

returnPackage=struct('fnamePrefix',fnamePrefix);
returnPackage.unfilteredData=transposedData;
returnPackage.filteredData=filteredData;
returnPackage.time=t;
returnPackage.sampleRate=sampleRate;
returnPackage.numChannels=numChannels;
returnPackage.noise=zeros(numChannels,1);
% returnPackage.adc=adcData;

% Sort out spikes
returnPackage.spikesGroup=cell(numChannels,1);
returnPackage.locationsGroup=cell(numChannels,1);
for traceIndex=1:numChannels
    % Prepare the data for each channel
    filteredChannel=filteredData(:,traceIndex)'; % This once again becomes a single row vector.
    numSelectedSortingRegion=size(filteredChannel,2);
    
    % Set threshold for detection of spikes
    switch thresholdMode
        case 'SetValue'
            threshold=thresholdSetValue;
        case 'Median'
            threshold=median(abs(filteredDataBeforeCAR(:,traceIndex)'))/0.6745*4;
    end
    returnPackage.noise(traceIndex)=threshold/4;
    
    % Locate the spike times
    switch detectMode
        case 'positive'
            index = find(filteredChannel(preSpike+clearRegion+1:end-postSpike-clearRegion) > threshold) +preSpike+clearRegion;
        case 'negative'
            index = find(filteredChannel(preSpike+clearRegion+1:end-postSpike-clearRegion) < -threshold) +preSpike+clearRegion;
        case 'both'
            index = find(abs(filteredChannel(preSpike+clearRegion+1:end-postSpike-clearRegion)) > threshold) +preSpike+clearRegion;
    end

    % Converge data points for one spike
    index_length=length(index);
    total_data_pts=preSpike+postSpike+1;
    spikes=zeros(index_length,total_data_pts);
    for i=1:index_length                          
        maxAbsValue=max(abs(filteredChannel(index(i)-preSpike-clearRegion:index(i)+postSpike+clearRegion)));
        if  maxAbsValue<artifactCutoff 
            switch detectMode
                case 'positive'
                    maxValue=0;
                    while max(filteredChannel(index(i)-preSpike:index(i)+postSpike))>maxValue && max(abs(filteredChannel(index(i)-preSpike:index(i)+postSpike)))<artifactCutoff && (index(i)-2*preSpike-clearRegion)>0 && (index(i)+2*postSpike+clearRegion)<=numSelectedSortingRegion
                        maxValue=max(filteredChannel(index(i)-preSpike:index(i)+postSpike));
                        index(i)=find(filteredChannel(index(i)-preSpike:index(i)+postSpike)==maxValue)+index(i)-preSpike-1;
                    end
                case 'negative'
                    minValue=0;
                    while min(filteredChannel(index(i)-preSpike:index(i)+postSpike))<minValue && max(abs(filteredChannel(index(i)-preSpike:index(i)+postSpike)))<artifactCutoff && (index(i)-2*preSpike-clearRegion)>0 && (index(i)+2*postSpike+clearRegion)<=numSelectedSortingRegion
                        minValue=min(filteredChannel(index(i)-preSpike:index(i)+postSpike));
                        index(i)=find(filteredChannel(index(i)-preSpike:index(i)+postSpike)==minValue)+index(i)-preSpike-1;
                    end
                case 'both'
                    maxValue=0;
                    while max(abs(filteredChannel(index(i)-preSpike:index(i)+postSpike)))>maxValue && max(abs(filteredChannel(index(i)-preSpike:index(i)+postSpike)))<artifactCutoff && (index(i)-2*preSpike-clearRegion)>0 && (index(i)+2*postSpike+clearRegion)<=numSelectedSortingRegion
                        maxValue=max(abs(filteredChannel(index(i)-preSpike:index(i)+postSpike)));
                        index(i)=find(abs(filteredChannel(index(i)-preSpike:index(i)+postSpike))==maxValue)+index(i)-preSpike-1;
                    end
            end
            maxClearRegion=max(max(abs(filteredChannel(index(i)-preSpike-clearRegion:index(i)-preSpike-1))),max(abs(filteredChannel(index(i)+postSpike+1:index(i)+postSpike+clearRegion))));
            mainPeak=filteredChannel(index(i));
            maxPostSpike=max(abs(filteredChannel(index(i)+postSpike/2:index(i)+postSpike)));
            maxPreSpike=max(abs(filteredChannel(index(i)-preSpike:index(i)-preSpike*0.4)));
            if mainPeak>0
                if maxClearRegion<mainPeak*0.5*2 && mainPeak<artifactCutoff && maxPostSpike<mainPeak*0.4 && maxPreSpike<mainPeak*0.8
                    spikes(i,:)=filteredChannel(index(i)-preSpike:index(i)+postSpike);
                end
            else
                if maxClearRegion<abs(mainPeak)*0.5*2 && abs(mainPeak)<artifactCutoff && maxPostSpike<abs(mainPeak)*0.4 && maxPreSpike<abs(mainPeak)*0.8
                    spikes(i,:)=filteredChannel(index(i)-preSpike:index(i)+postSpike);
                end
            end
        end
    end
    artifacts = find(spikes(:,preSpike)==0);       % Erase indices that were artifacts
    spikes(artifacts,:)=[];
    index(artifacts)=[];

    % Clean up redundancy
    [index sortNumber]=sort(index);
    spikes=spikes(sortNumber,:);
    currentIndex=0;
    toRemoveIndex=[];
    for i=1:size(index,2)
        if abs(index(i)-currentIndex)<postSpike+clearRegion
            toRemoveIndex=[toRemoveIndex i];
        else
            currentIndex=index(i);
        end
    end
    spikes(toRemoveIndex,:)=[];
    index(toRemoveIndex)=[];
    
    [numSpikes numDataPts]=size(spikes);
    
    returnPackage.spikeTimeStamp=([1:numDataPts]*1000/sampleRate)';
    returnPackage.spikesGroup{traceIndex}=spikes';
    returnPackage.indicesGroup{traceIndex}=index';
end
