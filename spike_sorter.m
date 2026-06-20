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
%set(0,'DefaultFigureWindowStyle','docked')
set(0,'DefaultFigureWindowStyle','normal')

%%%%%%%%%%%%%%%%%%%%%%%%%%PARAMETER INITIALIZATION%%%%%%%%%%%%%%%%%%%%%%%%%
% Multi-channel plot offsets uV
unfilteredPlotOffset=2000;
filteredPlotOffset=50;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%99%%%%%%%%%%%%%%%%%%FILE I/O%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File input
[fname pname]=uigetfile('*.*','Please select the Intan file to open');

% File output
pnameToSave=uigetdir(pname,'Please select the folder to save the processed files');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
results=spike_sorter_core(fname, pname,0,0);%time_change

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%LOADING FIELDS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fnamePrefix=results.fnamePrefix;
unfilteredData=results.unfilteredData;
filteredData=results.filteredData;
time=results.time;
sampleRate=results.sampleRate;
numChannels=results.numChannels;
spikeTimeStamp=results.spikeTimeStamp;
spikesGroup=results.spikesGroup;
indicesGroup=results.indicesGroup;
noise=results.noise;
% adcData=results.adc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% cd(pnameToSave);
% unfilteredFileName=strcat(fnamePrefix,'_unfiltered.dat');
% filteredFileName=strcat(fnamePrefix,'_filtered.dat');
% noiseFileName=strcat(fnamePrefix,'_noise.dat');
% 
%comment out the 5 line chunk below to save time; this exports unfiltered
%and filtered data

%  command=sprintf('save %s unfilteredData -ascii', unfilteredFileName);
%  eval(command);
%  command=sprintf('save %s filteredData -ascii', filteredFileName);
%  eval(command);
%  command=sprintf('save %s noise -ascii', noiseFileName);
%  eval(command);

% for traceIndex=1:numChannels
%     figure
%     hold on
%     for i=1:1:size(spikesGroup{traceIndex},2)
%         if filteredData(indicesGroup{traceIndex}(i),traceIndex)>0
%             plot(spikeTimeStamp,spikesGroup{traceIndex}(:,i),'r-','Linewidth',1);
%         else plot(spikeTimeStamp,spikesGroup{traceIndex}(:,i),'b-','Linewidth',1);
%         end
%         hold on
%     end
%     xlabel('Time [ms]','fontsize',36,'FontName','Arial','FontWeight','bold')
%     ylabel('Voltage [uV]','fontsize',36,'FontName','Arial','FontWeight','bold')
%     set(gca,'FontSize',24,'Linewidth',1,'box','off')
%     axis([0 3 -500 500])
%     toSaveMat=[spikeTimeStamp spikesGroup{traceIndex}];
%     indexStr=num2str(traceIndex);
%     while size(indexStr,2)<2
%         indexStr=strcat('0',indexStr);
%     end
%     toSaveName=strcat('Spikes_Channel_',indexStr,'.dat');
%     command=sprintf('save %s toSaveMat -ascii', toSaveName);
%     eval(command);
%     
%     peakLocationName=strcat('Spike_Locations_Channel_',indexStr,'.dat');
%     peakLocation=indicesGroup{traceIndex}/sampleRate+time(1)-1/sampleRate;
%     peakIndicator=[peakLocation filteredData(indicesGroup{traceIndex},traceIndex)+((filteredData(indicesGroup{traceIndex},traceIndex)>0)-0.5)*100];
%     command=sprintf('save %s peakLocation -ascii', peakLocationName);
%     eval(command);
%     figure 
%     plot(time,filteredData(:,traceIndex),'k-','Linewidth',1);
%     hold on
% %     plot(peakIndicator(find(peakIndicator(:,2)>0),1),peakIndicator(find(peakIndicator(:,2)>0),2),'r*');
% %     plot(peakIndicator(find(peakIndicator(:,2)<0),1),peakIndicator(find(peakIndicator(:,2)<0),2),'b*');
%     axis([time(1) time(size(time,1)) -500 500]) 
%     xlabel('Time [s]','fontsize',36,'FontName','Arial','FontWeight','bold')
%     ylabel('Voltage [uV]','fontsize',36,'FontName','Arial','FontWeight','bold')
%     set(gca,'FontSize',24,'Linewidth',2,'box','off')
% end
% 
% Plot all original, unfiltered traces

figure
for i=1:numChannels
    plot(time,unfilteredData(:,i)+unfilteredPlotOffset*(i-1),'k-');
    hold on
    axis([time(1) time(size(time,1)) -3*unfilteredPlotOffset unfilteredPlotOffset*(numChannels-1)+unfilteredPlotOffset*3])
end
set(gca,'FontSize',14,'Linewidth',1)
xlabel('Time [s]','fontsize',18,'FontName','Arial','FontWeight','bold')
ylabel('Voltage [uV]','fontsize',18,'FontName','Arial','FontWeight','bold')
title('Original Data')
set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
set(gcf,'color','white')

% figure
% for i=2
% y = fft(unfilteredData(:,i));
% %Plot the power spectrum as a function of frequency. While noise disguises a signal's frequency components in time-based space, the Fourier transform reveals them as spikes in power.
% 
% n = length(time);          % number of samples
% f = (0:n-1)*(20000/n);     % frequency range
% power = abs(y).^2/n;    % power of the DFT
% 
% plot(f,power)
% 
% end
% set(gca,'FontSize',14,'Linewidth',1)
% xlabel('Frequency')
% ylabel('Power')
% xlim([0 10]);


%figure('Units','normalized','Position',[0 0 1/3 1])
figure
for i=1:numChannels
    hold on
    plot(time,filteredData(:,i)+filteredPlotOffset*(i-1),'k-');
    axis([time(1) time(size(time,1)) -3*filteredPlotOffset filteredPlotOffset*(numChannels-1)+filteredPlotOffset*3])
end
%plot(time,(adcData/16500)*10^6-250,'k-');
set(gca,'FontSize',14,'Linewidth',1)
xlabel('Time [s]','fontsize',18,'FontName','Arial','FontWeight','bold')
ylabel('Voltage [uV]','fontsize',18,'FontName','Arial','FontWeight','bold')
set(gcf, 'units','normalized','outerposition',[0 0 1/4 1]);
set(gcf,'color','white')
%title('Filtered Data')
axis off

