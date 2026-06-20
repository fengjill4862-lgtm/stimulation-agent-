histSize=1; % ms
numBars=1000/histSize; % 1000 ms in total

[fnameL pnameL]=uigetfile('*.*','Get spike location file');
cd(pnameL)
spikeLocations=load(fnameL);

spikeIntervals=diff(spikeLocations)*1000; % in ms
x=histSize:histSize:histSize*numBars;
y=hist(spikeIntervals, x);
histPlot=[x' y'/sum(y)];
figure
bar(histPlot(1:numBars,1),histPlot(1:numBars,2)); % Only plot the first 500 ms
save spikeISI.dat histPlot -ascii