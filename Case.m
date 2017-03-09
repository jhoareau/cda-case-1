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

return

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
lambda_list = logspace(-5, 0, 100);
N_lambda = size(lambda_list, 2);
MSE = zeros(CV_folds,N_alpha,N_lambda);
Models = zeros(102,N_alpha,N_lambda);
for i=1:CV_folds
    i_alpha=1;
    Y_tr = Y_1(CV_indexes~=i); Y_te = Y_1(CV_indexes==i);
    X_tr = X_1(CV_indexes~=i,:); X_te = X_1(CV_indexes==i,:);
    
    for alpha=alpha_list
        i_lambda=1;
        for lambda=lambda_list
            [B,FitInfo] = lasso(X_tr, Y_tr, 'Alpha', alpha, 'Lambda', lambda);
            MSE(i,i_alpha,i_lambda) = MSE(i,i_alpha,i_lambda) + FitInfo.MSE;
            Models(:,i_alpha,i_lambda) = B;
            i_lambda=i_lambda+1;
        end
        i_alpha=i_alpha+1;
    end
end

MSE = reshape(mean(MSE,1), [size(MSE, 2) size(MSE, 3)]);
heatmap(MSE, lambda_list, alpha_list, '%0.3e', 'Colorbar', true, 'UseLogColormap', true);