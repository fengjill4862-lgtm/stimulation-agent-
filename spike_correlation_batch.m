
tic
[fname pname]=uigetfile('*.*','Please select the Intan file to open');
cd(pname);
filenamelist=dir(pwd);
nel=length(filenamelist)-2;
% File outputw
pnameToSave=uigetdir(pname,'Please select the folder to save the processed files');

tableCoeff=zeros([nel nel]);
filenames = strings(1,nel);
match = ["Locations_" ".dat"];

for i=1:nel
    currentFname1=filenamelist(i+2).name;
    for j=i+1:nel
        currentFname2=filenamelist(j+2).name;
        spike_correlation_batch_core(currentFname1, currentFname2, pname, pnameToSave);
        % newresults=spike_correlation_batch_core_Jongha(currentFname1, currentFname2, pname, pnameToSave);
        %tableCoeff(i,j)=newresults(1);
        %tableCoeff(j,i)=newresults(1);
    end
    filenames(i)=erase(filenamelist(i+2).name, match);
end
toc