
function results=spike_correlation_batch_core(fname1,fname2,pname1,pname2);

match = ["Locations_" ".dat"];
filename1 = erase(fname1, match);
filename2 = erase(fname2, match);
underscore_indices1 = strfind(filename1,'_'); 
fnametitle1 = filename1([underscore_indices1(end-1)+1:end]);
underscore_indices2 = strfind(filename2,'_'); 
fnametitle2 = filename2([underscore_indices2(end-1)+1:end]);
savename=fnametitle1+"+"+fnametitle2+".jpg";

%%%%%%%%%% Parameter Initialization%%%%%%%%%%%%%
sampleRate=20000;
binningFactor=40; % binningFactor/sampleRate = binTime
sampleSize=10000;
%numAverages=10000;

% File I/O
cd(pname1);
spike_locations_1 = load(fname1);
trace_1 = zeros(round(max(spike_locations_1)*sampleRate),1);
trace_1(nonzeros(round(spike_locations_1*sampleRate)),1) = 1;

spike_locations_2 = load(fname2);
trace_2 = zeros(round(max(spike_locations_2)*sampleRate),1);
trace_2(nonzeros(round(spike_locations_2*sampleRate)),1) = 1;

traceNumPts=min(size(trace_1,1),size(trace_2,1));
%toAverageMatrix=zeros(numAverages,2*sampleSize-1);
% for startIndex=1:1:10000
%     startIndex
%     [coeff lag] = xcorr(trace_1(startIndex:startIndex+sampleSize-1),trace_2(startIndex:startIndex+sampleSize-1));
%     toAverageMatrix(startIndex,:)=coeff';
% end
[coeff lag] = xcorr(trace_1,trace_2,20000);
%newCoeff=mean(toAverageMatrix)';
newCoeff=coeff;
numBins=floor(size(lag,2)/binningFactor);
newCoeff=sum(reshape(newCoeff(1:numBins*binningFactor,:),[binningFactor numBins]));
newLag=reshape(lag(:,1:numBins*binningFactor),[binningFactor numBins]);
newLag=newLag(1,:);

%results=[0 0];

figure('visible','off')
bar(newLag/sampleRate,newCoeff)
toSave=[newLag'/sampleRate newCoeff'];
axis([-0.1 1 0 inf]);
set(gca,'FontSize',16,'Linewidth',2,'XLim',[-0.1 0.1],'box','off')
xlabel('Time [s]','fontsize',16,'FontName','Arial','FontWeight','bold')
ylabel('Correlation Index [a.u.]','fontsize',16,'FontName','Arial','FontWeight','bold')
title([fnametitle1 '+' fnametitle2],'Interpreter', 'none');

cd(pname2);
saveas(gcf, savename);

[coeff2 lag2] = xcorr(trace_2,trace_1,20000);
newCoeff2=coeff2;
numBins2=floor(size(lag2,2)/binningFactor);
newCoeff2=sum(reshape(newCoeff2(1:numBins2*binningFactor,:),[binningFactor numBins2]));
newLag2=reshape(lag2(:,1:numBins2*binningFactor),[binningFactor numBins2]);
newLag2=newLag2(1,:);

figure('visible','off')
bar(newLag2/sampleRate,newCoeff2)
toSave=[newLag2'/sampleRate newCoeff2'];
axis([-0.1 0.1 0 inf]);
set(gca,'FontSize',16,'Linewidth',2,'XLim',[-0.1 0.1],'box','off')
xlabel('Time [s]','fontsize',16,'FontName','Arial','FontWeight','bold')
ylabel('Correlation Index [a.u.]','fontsize',16,'FontName','Arial','FontWeight','bold')
title([fnametitle2 '+' fnametitle1],'Interpreter', 'none');


%results(1)=mean(toSave(994:998,2))/mean(toSave(901:951,2));
%results(2)=mean(toSave(1003:1007,2))/mean(toSave(1051:1101,2));

%results(1)=A2Bstrength;
%results(2)=B2Astrength;
savename2=fnametitle2+"+"+fnametitle1+".jpg";


cd(pname2);
saveas(gcf, savename2);
end