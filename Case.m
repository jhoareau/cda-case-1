clc; clear all; close all;
data = readtable('Case1_Data.xls');

N_samples = size(data, 1);
N_columns = size(data, 2) - 1;

X = data(:, 2:N_columns+1);

%% Coding the last column
X100 = table2array(X(:, end));
X = table2array(X(:, 1:end-1));
X100_new = double(strcmp(X100, 'A'));
X101 = double(strcmp(X100, 'B'));
X102 = double(strcmp(X100, 'C'));
X = [ X, X100_new, X101, X102 ];

y = table2array(data(:, 1));

N_columns = size(X, 2);

%% Boxplotting the data
boxplot(X(:, 1:end-3), 1:N_columns-3);

%% Clearing missing data
% Replacing NaNs with the mean of the rest of the column
for j=1:N_columns
    % mean_col = nanmedian(data(:, j));
    mean_col = nanmean(X(:, j));
    for i=1:N_samples
        if isnan(X(i, j))
            X(i,j) = mean_col;
        end
    end
end

%% Training data itself
X_1 = X(1:99, :);
y_1 = y(1:99, :);

%% Linear regression
CV_folds = 10;
CV_indexes = crossvalind('Kfold', N_samples, CV_folds);
for i=1:CV_folds
    Y_train = Y(CV_indexes~=i); Y_test = Y(CV_indexes==i);
    X_tr = X(CV_indexes~=i,:); X_te = X(CV_indexes==i,:);
    
    [X_train, moy_x, var_x] = normalize(Xtr);
    X_test = normalizetest(X_te, moy_x, var_x);
    
    
end