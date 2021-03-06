clc; clear all; close all;
data = readtable('Case1_Data.xls');

N_samples_total = size(data, 1);
N_samples_train = 99;
N_columns = size(data, 2) - 1;

X = data(:, 2:N_columns+1);

%% Coding the last column
X100 = table2array(X(:, end));
X = table2array(X(:, 1:end-1));
X100_new = double(strcmp(X100, 'A'));
X101 = double(strcmp(X100, 'B'));
X102 = double(strcmp(X100, 'C'));
X = [ X, X100_new, X101, X102 ];

Y = table2array(data(:, 1));

N_columns = size(X, 2);
clearvars X100 X100_new X101 X102

%% Boxplotting the data
boxplot(X(:, 1:end-3), 1:N_columns-3);

%% Clearing missing data
% Replacing NaNs with the mean of the rest of the column
for j=1:N_columns
    % mean_col = nanmedian(data(:, j));
    mean_col = nanmean(X(1:N_samples_train, j));
    for i=1:N_samples_total
        if isnan(X(i, j))
            X(i,j) = mean_col;
        end
    end
end

%% Training data itself
X_1 = X(1:N_samples_train, :);
Y_1 = Y(1:N_samples_train, :);

X_2 = X(N_samples_train+1:end, :);
Y_2 = Y(N_samples_train+1:end, :);

%% Linear regression
[X_train, moy_x, var_x] = normalize(X_1);
[Y_train, moy_y] = center(Y_1);

B = regress(Y_1, X_train);
Y_est = X_train*B;

% Predictions for the rest of the data
X_test = normalizetest(X_2, moy_x, var_x);
Y_predicted = X_test*B + moy_y;

RSS = sum((Y_train-Y_est).^2);
MSE = mean((Y_train-Y_est).^2);
TSS = sum((Y_train-mean(Y_est)).^2);
R2 = 1 - RSS / TSS;

disp('LINEAR REGRESSION RESULTS');
disp(['MSE: ' num2str(MSE)]);
disp(['R2: ' num2str(R2)]);

% Considering there are more parameters than observations
% Linear regression is only overfitting the training data.
% We cannot use it at all to predict for the test data.

%% Ridge regression
lambda = logspace(-4,3,100);
Beta = zeros(102,100);
for i=1:100
    Beta(:,i) = (X_1'*X_1+lambda(i)*eye(102)) \ X_1'*Y_1;
end
figure(1);
semilogx(lambda,Beta')
xlabel('\lambda')
ylabel('\beta estimates')
title('Regularized estimates')

N = 99;
CV_folds = 10;
CV_indexes = crossvalind('Kfold', N_samples_train, CV_folds);
lambda = logspace(-4,3,100);
for i=1:CV_folds
    Ytrain = Y_1(CV_indexes~=i); Ytest = Y_1(CV_indexes==i);
    Xtrain = X_1(CV_indexes~=i,:); Xtest = X_1(CV_indexes==i,:);
    [Ytrain,my] = center(Ytrain); 
    Ytest = Ytest-my; %center the test response
    [Xtrain,mx,varx] = normalize(Xtrain); % normalize training data
    Xtest=normalizetest(Xtest,mx,varx); % normalize test data with mean and 
                                      % variance of training data
    for j=1:100
        Beta = (Xtrain'*Xtrain+lambda(j)*eye(102)) \ Xtrain'*Ytrain;
        MSE(i,j)= mean((Ytest-Xtest*Beta).^2); %MSE lambda
    end
end

meanMSE = mean(MSE);
[tmp j_opt] = min(meanMSE);
Lambda_CV = lambda(j_opt);
hold on
semilogx([Lambda_CV Lambda_CV],[-10 10],'--r')
hold off
disp(['CV lambda = ',num2str(Lambda_CV)]);
disp(['MSE lambda = ',num2str(tmp)]);

seMSE = std(MSE)/sqrt(CV_folds);

J = find(meanMSE(j_opt) + seMSE(j_opt) > meanMSE);
j = J(end);
Lambda_CV_1StdErrRule = lambda(j); 
hold on
semilogx([Lambda_CV_1StdErrRule Lambda_CV_1StdErrRule],[-10 10],'--b')
hold off
disp(['CV lambda 1-std-rule = ',num2str(Lambda_CV_1StdErrRule)]);
disp(['MSE lambda 1-std-rule = ',num2str(meanMSE(j))]);
%disp(seMSE)

%% Lasso regression: LARS
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
    for j=1:K
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

%% ElasticNet

CV_folds = 10;
CV_indexes = crossvalind('Kfold', N_samples_train, CV_folds);

alpha_list = 0.0001:0.1:1;
N_alpha = size(alpha_list, 2);
lambda_list = logspace(-4, 3, 100);
N_lambda = size(lambda_list, 2);
MSE = zeros(CV_folds,N_alpha,N_lambda);
Models = zeros(size(X,2),N_alpha,N_lambda);
for i=1:CV_folds
    i_alpha=1;
    Y_tr = Y_1(CV_indexes~=i); Y_te = Y_1(CV_indexes==i);
    X_tr = X_1(CV_indexes~=i,:); X_te = X_1(CV_indexes==i,:);
    
    for alpha=alpha_list
        i_lambda=1;
        for lambda=lambda_list
            [B,FitInfo] = lasso(X_tr, Y_tr, 'Alpha', alpha, 'Lambda', lambda);
            YhatTst = X_te*B;
            MSE(i,i_alpha,i_lambda) = MSE(i,i_alpha,i_lambda) + (YhatTst-Y_te)'*(YhatTst-Y_te)/length(Y_te);
            i_lambda=i_lambda+1;
        end
        i_alpha=i_alpha+1;
    end
end

MSE = reshape(mean(MSE,1), [size(MSE, 2) size(MSE, 3)])/sqrt(CV_folds);