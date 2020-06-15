function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));


% cost
rg_theta = (lambda / 2) * sum(sum(Theta .* Theta));
rg_x = (lambda / 2) * sum(sum(X .* X));
J = (1/2)*sum(sum(((X * Theta' - Y) .* R).^2)) + rg_theta + rg_x;

for i=[1:size(Theta)(2)]
  Theta_k = Theta(:,i);
  X_k = X(:,i);
  % gradient X
  X_grad(:, i) = sum(((X * Theta' - Y) .* Theta_k') .* R, 2) + lambda * X_k;
  % gradient Theta
  Theta_grad(:, i) = sum((((X * Theta' - Y) .* X_k) .* R)', 2) + lambda * Theta_k;
end

grad = [X_grad(:); Theta_grad(:)];

end
