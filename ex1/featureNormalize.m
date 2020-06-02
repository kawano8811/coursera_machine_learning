function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
       
mu = mean(X);
sigma = std(X);
X_norm(:,1) = (X(:,1) - mu(:, 1)) / sigma(:,1);
X_norm(:,2) = (X(:,2) - mu(:, 2)) / sigma(:,2);







% ============================================================

end
