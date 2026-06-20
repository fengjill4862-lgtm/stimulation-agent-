

% Original .m file takes both the wave_clus input file (.mat file, containing a
% .spikes field for all spike waveforms and a .index field for the spike
% locations) and the wave_clus output file (also in .mat format, which is
% the file after wave_clus clustering, containing the most important field
% of cluster_class, which labels all spike indices with cluster number).
% Then this .m file generates the scatter files for all clusters in the
% PC1-PC2 plane and the individualized spike location files for all
% clusters as well. 

% Modified version
% (1) load all channels at the same folder
% Output files = Spikes, average spikes, scatter, locations in all channels

clear
close all
set(0,'DefaultFigureWindowStyle','docked')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%SET PARAMETERS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numClusters=5; % (1) set the cluster number
spikeAxis = [0 3.1 -50 30]; %(2) set the spike axis, important for the data image
barScale = [50 1]; %in uV and ms [50 1] == 50uV and 1ms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%UI interface%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
axisONOFF='no'; %(5) set the axis to be shown; 'yes' or 'no'
plotMode='stack'; % (6) average or stack; average plot only the average
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%SET PARAMETERS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[oriFname oriPname]=uigetfile('*.*','Get the original file');
cd(oriPname);
[proFname proPname]=uigetfile('*.*','Get processed file');
savPname=uigetdir(proPname,'Please select the DIFFERENT folder to save the processed files');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%(1) PLOT PCA%%%%%%%%%%%%%%%%%%%%%%%%%
    cd(oriPname);
    oriData=load(oriFname);
    cd(proPname);
    proData=load(proFname);
    
    [coeff score]=pca(oriData.spikes);

    % Change PC indices here
    PCIndex1=1;
    PCIndex2=2;
    
    forScatterPlot=zeros(length(score),2);
    forScatterPlot(:,1)=score(:,PCIndex1);
    forScatterPlot(:,2)=score(:,PCIndex2);

    colorOptions=['r' 'g' 'b' 'c' 'y' 'k'];
    %colorOptions=['g' 'r' 'c' 'b' 'y' 'k'];
    
    figure
    scatterFiles=cell(1,numClusters);
    locationFiles=cell(1,numClusters);
    waveformFiles=cell(1,numClusters);
    
    for i=1:numClusters
%     for  i = 2
        scatterFiles{1,i}=[forScatterPlot(find(proData.cluster_class(:,1)==i),1) forScatterPlot(find(proData.cluster_class(:,1)==i),2)];
        locationFiles{1,i}=proData.cluster_class(find(proData.cluster_class(:,1)==i),2);
        waveformFiles{1,i}=oriData.spikes(find(proData.cluster_class(:,1)==i),:)';
        scatter(forScatterPlot(find(proData.cluster_class(:,1)==i),1),forScatterPlot(find(proData.cluster_class(:,1)==i),2),600,strcat(colorOptions(i),'.'));
        hold on
    end
    
    set(gca,'FontSize',30,'Linewidth',2,'box','off')
    xlabel('PC1','fontsize',30,'FontName','Arial','FontWeight','bold')
    ylabel('PC2','fontsize',30,'FontName','Arial','FontWeight','bold')
    title(['Ch PCA'], 'fontsize',30,'FontName','Arial','FontWeight','bold')
    cd(savPname);
    set(gcf, 'renderer', 'Painters')
    saveas(gcf,['PCA_Ch.jpg']);
    hold off;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%(2) PLOT SPIKES%%%%%%%%%%%%%%%%%%%%%%%%%
      avgredSpikes= [transpose([0:0.05:3]) mean(waveformFiles{1,1},2) std(waveformFiles{1,1},0, 2)];
      avggreenSpikes= [transpose([0:0.05:3]) mean(waveformFiles{1,2},2) std(waveformFiles{1,2},0, 2)];
      avgblueSpikes= [transpose([0:0.05:3]) mean(waveformFiles{1,3},2) std(waveformFiles{1,3},0, 2)];
     avgcyanSpikes= [transpose([0:0.05:3]) mean(waveformFiles{1,4},2) std(waveformFiles{1,4},0, 2)];
    avgyellowSpikes= [transpose([0:0.05:3]) mean(waveformFiles{1,5},2) std(waveformFiles{1,5},0, 2)];
    %avgblackSpikes= [transpose([0:0.05:3]) mean(waveformFiles{1,6},2) std(waveformFiles{1,6},0, 2)];
          
    switch plotMode
        case 'average'
            figure ('visible', 'on')
            for i=1:numClusters 
                plot([0:0.05:3], mean(waveformFiles{1,i},2), strcat(colorOptions(i),'-'), 'Linewidth', 15);
                hold on                
            end
             
        case 'stack'
            for i=1:numClusters
                figure ('visible', 'on') %to plot clusters in seperate figures
                %for j=1:size(waveformFiles{1,i},2)
                for j=1:50 % plot 50 spikes
                plot([0:0.05:3],waveformFiles{1,i}(:,j),strcat(colorOptions(i),'-'),'Linewidth',2);
                hold on
                end
                hold off %to plot clusters in seperate figures
                set(gcf, 'renderer', 'Painters')
                axis(spikeAxis);
                 box off
                 axis off

            end
    end
        
    switch axisONOFF
        case 'yes'
            set(gca,'FontSize',30,'Linewidth',2,'box','off')
            xlabel('Time [ms]','fontsize',30,'FontName','Arial','FontWeight','bold')
            ylabel('Voltage [uV]','fontsize',30,'FontName','Arial','FontWeight','bold')    
            title(['Ch' ' SortedSpikes'], 'fontsize',30,'FontName','Arial','FontWeight','bold')
    
        case 'no'
            box off
            axis off
    end
	
    set(gcf, 'renderer', 'Painters')
    axis(spikeAxis);
%     saveas(gcf,['TotSpikes_Ch.emf']);
%     saveas(gcf,['TotSpikes_Ch.jpg']);
        
    redSpikes=waveformFiles{1,1};
    greenSpikes=waveformFiles{1,2};
    blueSpikes=waveformFiles{1,3};
    cyanSpikes=waveformFiles{1,4};
    yellowSpikes=waveformFiles{1,5};
%     blackSpikes=waveformFiles{1,6};
    
    redScatter=scatterFiles{1,1};
    greenScatter=scatterFiles{1,2};
    blueScatter=scatterFiles{1,3};
    cyanScatter=scatterFiles{1,4};
    yellowScatter=scatterFiles{1,5};
%     blackScatter=scatterFiles{1,6};
    
    redLocations=locationFiles{1,1}/1000;
    greenLocations=locationFiles{1,2}/1000;
    blueLocations=locationFiles{1,3}/1000;
    cyanLocations=locationFiles{1,4}/1000;
    yellowLocations=locationFiles{1,5}/1000;
%     blackLocations=locationFiles{1,6}/1000;
    
    cd(savPname);
    
    save(['Spikes_Ch_red.dat'], 'redSpikes', '-ascii');
    save(['PCI_Ch_red.dat'], 'redScatter', '-ascii');
    save(['Locations_Ch_red.dat'], 'redLocations', '-ascii');
    save(['AvgSpikes_Ch_red.dat'], 'avgredSpikes', '-ascii');
    
    save(['Spikes_Ch_green.dat'], 'greenSpikes', '-ascii');
    save(['PCI_Ch_green.dat'], 'greenScatter', '-ascii');
    save(['Locations_Ch_green.dat'], 'greenLocations', '-ascii');
    save(['AvgSpikes_Ch_green.dat'], 'avggreenSpikes', '-ascii');
    
    save(['Spikes_Ch_blue.dat'], 'blueSpikes', '-ascii');
    save(['PCI_Ch_blue.dat'], 'blueScatter', '-ascii');
    save(['Locations_Ch_blue.dat'], 'blueLocations', '-ascii');
    save(['AvgSpikes_Ch_blue.dat'], 'avgblueSpikes', '-ascii');
    
    save(['Spikes_Ch_cyan.dat'], 'cyanSpikes', '-ascii');
    save(['PCI_Ch_cyan.dat'], 'cyanScatter', '-ascii');
    save(['Locations_Ch_cyan.dat'], 'cyanLocations', '-ascii');
    save(['AvgSpikes_Ch_cyan.dat'], 'avgcyanSpikes', '-ascii');
    
    save(['Spikes_Ch_yellow.dat'], 'yellowSpikes', '-ascii');
    save(['PCI_Ch_yellow.dat'], 'yellowScatter', '-ascii');
    save(['Locations_Ch_yellow.dat'], 'yellowLocations', '-ascii');
    save(['AvgSpikes_Ch_yellow.dat'], 'avgyellowSpikes', '-ascii');
%     
%     save(['Spikes_Ch_black.dat'], 'blackSpikes', '-ascii');
%     save(['PCI_Ch_black.dat'], 'blackScatter', '-ascii');
%     save(['Locations_Ch_black.dat'], 'blackLocations', '-ascii');
%     save(['AvgSpikes_Ch_black.dat'], 'avgblackSpikes', '-ascii');
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%(3) PLOT SCALE BAR%%%%%%%%%%%%%%%%%%%%%%%%%
figure('visible','off')
plot([1 1+barScale(2)],[0 0], 'Linewidth', 15, 'color', 'k');
hold on
plot([1 1], [0 barScale(1)], 'Linewidth', 15, 'color', 'k');
axis(spikeAxis);
hold off
axis off
box off
set(gcf, 'renderer', 'Painters')
saveas(gcf,'TotScalebar.jpg');
cd(savPname);
saveas(gcf,'TotScalebar.emf');