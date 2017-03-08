clc; clear all; close all;
% Least angle regression and selection LARS 
load larsdata
CV_folds = 10;
CV_indexes = crossvalind('Kfold', N_samples_train, CV_folds);

Err_tr = [];
 Err_test =[];
for i=1:CV_folds
    Ytr = Y_1(CV_indexes~=i); Ytst = Y_1(CV_indexes==i);
    Xtr = X_1(CV_indexes~=i,:); Xtst = X_1(CV_indexes==i,:);
    [Ytr,my] = center(Ytr); 
    Ytst = Ytst-my; %center the test response
    [Xtr,mx,varx] = normalize(Xtr); % normalize training data
    Xtst=normalizetest(Xtst,mx,varx); % normalize test data with mean and variance of training data
    Beta = lar(Xtr,Ytr);
    K=size(Beta,2);
    for j=1:K;
       beta = Beta(:,j);
       YhatTr = Xtr*beta;
       YhatTst = Xtst*beta;
       Err_tr(i,j) = (YhatTr-Ytr)'*(YhatTr-Ytr)/length(Ytr);
       Err_test(i,j) = (YhatTst-Ytst)'*(YhatTst-Ytst)/length(Ytst);
    end
end
err_tr = mean(Err_tr,1);
err_test = mean(Err_test,1);
err_ste = std(Err_test,1)/sqrt(CV_folds); 

%plot
figure
plot(1:K+1,err_tr,'-b'), hold on
errorbar(1:K+1,err_test,err_ste,'r')
legend('train','test')

xlabel('k')
ylabel('error estimate')

