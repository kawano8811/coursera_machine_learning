function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

h = (X * theta); % hypothesis

tj = theta(2:size(theta)(1)); % theta for J, exclude bias
rj = (lambda / (2 * m)) * sum(tj .* tj); % regularization for J
J = (1 / (2*m)) * sum(power(h - y, 2)) + rj; % cost

tg = theta;
tg(1) = 0; % theta for grad, bias is 0
rg = (lambda / m) * tg; % regularization for grad
grad = ((1 / m) * sum((h - y) .* X))' + rg;


grad = grad(:);

end
