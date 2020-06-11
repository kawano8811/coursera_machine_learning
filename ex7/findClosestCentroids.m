function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

J = zeros(size(X,1), K); % distances of all centroids
for i=[1:K]
  cent = centroids(i,:); % k-th centroid
  j = sum((X - cent).^2, 2); % distance from centroid
  J(:,i) = j;
endfor

% closest centroid index
[d, idx] = min(J, [], 2);

end

