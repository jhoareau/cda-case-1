function R = choldelete(R,j)
%CHOLDELETE Fast downdate of Cholesky factorization of X'*X.
%   CHOLDELETE returns the Cholesky factorization of the Gram matrix X'*X
%   where the jth column of X has been removed.
%
%   R = CHOLDELETE(R, j) returns a matrix corresponding to R =
%   chol(X2'*X2), where X2 is equal to X with the jth column taken out and
%   R = chol(X'*X) is the Cholesky factorization of X'*X to be downdated.
%
%   This function is an auxiliary part of SpaSM [1], a matlab toolbox for
%   sparse modeling and analysis.
%
%   References
%   -------
%   [1] K. Sj�strand, L.H. Clemmensen, M. M�rup. SpaSM, a Matlab Toolbox
%   for Sparse Analysis and Modeling. Journal of Statistical Software
%   x(x):xxx-xxx, 2010.
%
%  See also CHOLINSERT.

R(:,j) = []; % remove column j
n = size(R,2);
for k = j:n
  p = k:k+1;
  [G,R(p,k)] = planerot(R(p,k)); % remove extra element in column
  if k < n
    R(p,k+1:n) = G*R(p,k+1:n); % adjust rest of row
  end
end
R(end,:) = []; % remove zero'ed out row
