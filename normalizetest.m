function [X] = normalizetest(X,mx,d)
% NORMALIZE  Normalize the observations of a test data matrix given the mean mx and varaince varx of the training.
%    X = NORMALIZE(X,mx,varx) centers and scales the observations of a data
%    matrix such that each variable (column) has unit length.
% returns:
% X: the normalized test data
%
% Edited: Line Clemmensen, IMM, DTU, lhc@imm.dtu.dk

n = size(X,1);
X=(X-ones(n,1)*mx)./(ones(n,1)*d);
