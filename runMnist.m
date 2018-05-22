%% fragment of MNIST script for the paper
%% train kmeans_l2 with initializatioin from first inputs
clearvars;close all
seed=rng(2018);
tic;
%% load data and ground truth
addpath('~/Documents/MNIST');
img0=loadMNISTImages('train-images-idx3-ubyte');%'train-images.idx3-ubyte');
label=loadMNISTLabels('train-labels-idx1-ubyte');
%% normalize, no centering
nrm=sqrt(sum(img0.^2));
img=img0./(ones(784,1)*nrm);
clear img0
%% test sample
img_t0=loadMNISTImages('t10k-images-idx3-ubyte');
label_t=loadMNISTLabels('t10k-labels-idx1-ubyte');
nrm_t=sqrt(sum(img_t0.^2));
img_t=img_t0./(ones(784,1)*nrm_t);
clear img_t0
%%
nlgalphas = [Inf 2.5]; %3:-.5:2
n_init=10;
for k=20 %[10,20,50,100,200]
for rep=1:3
    %% init
    iniperm=randperm(n_init*k);
    winit=zeros(784,k);
    for i=0:(k-1)
        winit(:,i+1) = mean( img(:,iniperm(n_init*i+(1:n_init))) ,2 )';
    end
    T=length(label-n_init*k);
    X=img(:,(n_init*k+1):end);
    digit=label((n_init*k+1):end);
    %% train test k-means
    sparsity=[];
    accuracy_t=[];
    for nlgalpha=nlgalphas
        alpha=10^-nlgalpha;
        %% train
        if alpha==0
            [Y,w,Z] = kmeans_neural( X, winit );
        else
            [Y,w,Z] = kmeans_L2_neural( X, alpha, winit );
        end
        %% test
        Yt = kmeans_L2_neural_noupd( img_t, alpha, w, sum(Y,2), (Y*Z')./sum(Y,2) );
        sparsity=[sparsity; sum(Yt(:)>0) / length(Yt(:)) ];
        %% multiclass svm: latest of training, then test
        latest=10000;
        scores=zeros(10,size(Yt,2));
        for d=0:9
            lsvm=fitclinear(Y(:,end-latest+1:end)',digit(end-latest+1:end)==d);%,...
        %         'OptimizeHyperparameters',{'lambda'},...
        %         'HyperparameterOptimizationOptions',struct('Verbose',0,'ShowPlots',false));
            [lsvmlbl,lsvmscore]=predict(lsvm,Yt');
            scores(d+1,:) = lsvmscore(:,2); %target class in 2nd column
        end
        [~,digitAssign] = max(scores);
        conftab=zeros(10,10);
        for i=1:length(label_t)
            conftab(digitAssign(i),label_t(i)+1) = conftab(digitAssign(i),label_t(i)+1) +1;
        end
        pctCorrect=100*sum(diag(conftab))/length(label_t);
        accuracy_t=[accuracy_t,pctCorrect];
    end
    %%
    disp(['k=' num2str(k) ' Rep ' num2str(rep) '  toc ' num2str(toc)])
    disp('Alpha  Sparsity  Accuracy')
    for ia=1:length(nlgalphas)
        disp([10^-nlgalphas(ia), sparsity(ia), accuracy_t(ia)])
    end
end
end
