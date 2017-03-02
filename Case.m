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

%% Lasso regression
CV_folds = 10;
CV_indexes = crossvalind('Kfold', N_samples_train, CV_folds);
MSE_array = zeros(N_samples_train, CV_folds);
for i=1:CV_folds
    Y_tr = Y_1(CV_indexes~=i); Y_te = Y_1(CV_indexes==i);
    X_tr = X_1(CV_indexes~=i,:); X_te = X_1(CV_indexes==i,:);
    
    [X_train, moy_x, var_x] = normalize(X_tr);
    [Y_train, moy_y] = center(Y_tr);
    X_test = normalizetest(X_te, moy_x, var_x);
    Y_test = Y_te - moy_y;
    
    B = regress(Y_train, X_train);
    Y_est = X_test*B;
    MSE_array(:, i) = sum((Y_est - Y_test).^2);
end
plot(1:CV_folds, MSE_array);