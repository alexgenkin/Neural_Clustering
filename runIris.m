%% Iris script for the paper
clearvars;
rng(2018);
load fisheriris
%%
names=unique(species);
[N,n]=size(meas);
class=zeros(N,1);
for i=1:N
    if strcmp(species{i},'setosa'), class(i)=1;
    elseif strcmp(species{i},'versicolor'), class(i)=2;
    elseif strcmp(species{i},'virginica'), class(i)=3;
    end
end
%%
%X=meas./(ones(n,1)*std(meas));
means=mean(meas);
Xc=meas-(ones(N,1)*mean(meas));
k=3;
seed=rng();
%%
ctKm=[];
ctNeural=[];
randKm=[];
randNeural=[];
rep=10;
%% loop
for i=1:rep
    winit=diag(std(Xc))*.5*randn(n,k);
    perm=randperm(N);
    Xcp=Xc(perm,:);
    classp=class(perm,:);
    [Y,w,wtrace]=kmeans_online(Xcp',winit);
    oidx=(1:k)*Y;
    r=randIndex(oidx,classp);
    disp(['k-means  rep ' num2str(i)   ' Rand=' num2str(r)])
    ct0=crosstab(oidx,classp);
    [~,ord]=max(ct0);
    ct=ct0(ord,:)'; %transpose: class on rows
    disp(ct)
    % neural
    [Yn,wn,z,wtracen]=kmeans_neural(Xcp',winit,'neural');
    Yn(Yn>0.99)=1;
    onidx=(1:k)*Yn;
    rn=randIndex(onidx,classp);
    disp(['neural  rep ' num2str(i)   ' Rand=' num2str(rn)])
    ctn0=crosstab(onidx,classp);
    [~,ordn]=max(ctn0);
    ctn=ctn0(ordn,:)'; %transpose: class on rows
    disp(ctn);

    %- collect
    if length(unique(ord))<3,continue;end
    if length(unique(ordn))<3,continue;end
    ctKm=[ctKm; ct(:)'];
    ctNeural=[ctNeural; ctn(:)'];
    randKm=[randKm;r];
    randNeural=[randNeural;rn];
end
%%
%%
ctKmM=reshape(mean(ctKm),3,3);
ctKmS=reshape(std(ctKm),3,3);
ctNeuralM=reshape(mean(ctNeural),3,3);
ctNeuralS=reshape(std(ctNeural),3,3);
disp 'confusion table k-means, mean and stdev'
disp([ctKmM  ctKmS])
disp 'confusion table neural, mean and stdev'
disp([ctNeuralM  ctNeuralS])
disp 'Rand Index: k-means mean,stdev; neural mean,stdev'
disp([mean(randKm) std(randKm)  mean(randNeural) std(randNeural)])
