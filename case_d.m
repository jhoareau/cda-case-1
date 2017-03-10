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

%% Boxplotting the data
%boxplot(X(:, 1:end-3), 1:N_columns-3);

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


%% Ridge regression

%a %%%%%%%%%%%%%%%%
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


%b %%%%%%%%%%%%%%%%
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
        SSE(i,j)= sum((Ytest-Xtest*Beta).^2);
    end
end
MSE2 = sum(SSE,1)/N;
meanMSE = mean(MSE); %
[tmp j_opt] = min(meanMSE);
Lambda_CV = lambda(j_opt);
hold on
semilogx([Lambda_CV Lambda_CV],[-10 10],'--r')
hold off
disp(['CV lambda = ',num2str(Lambda_CV)]);


%c - 1stderror %%%%%%%%%%%%%%%%
seMSE = std(MSE)/sqrt(CV_folds);

J = find(meanMSE(j_opt) + seMSE(j_opt) > meanMSE);
j = J(end);
Lambda_CV_1StdErrRule = lambda(j); 
hold on
semilogx([Lambda_CV_1StdErrRule Lambda_CV_1StdErrRule],[-10 10],'--b')
hold off
disp(['CV lambda 1-std-rule = ',num2str(Lambda_CV_1StdErrRule)]);
%disp(seMSE)


