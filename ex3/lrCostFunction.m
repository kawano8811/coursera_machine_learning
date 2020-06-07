function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
h = sigmoid(X * theta);


tj = theta(2:size(theta)(1)); % theta for J, exclude bias
tg = theta;
tg(1) = 0; % theta for grad, bias is 0

rj = (lambda / (2 * m)) * sum(tj .* tj); % regularization for J
rd = (lambda / m) * tg;

J = (1 / m) * sum(-y .* log(h) - (1 - y) .* log(1 - h)) + rj;
grad = ((1 / m) * sum((h - y) .* X))' + rd;

end
