function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
                 
% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Feedforward
a1 = [ones(m, 1) X]; % add bias

z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2]; % add bias

z3 = a2 * Theta2';
a3 = sigmoid(z3);
h = a3;

% Cost Function
y_m = [zeros(m, num_labels)];
for i=[1:m]
  y_m(i, y(i)) = 1;
end
J = (1/m) * sum(sum((-y_m .* log(h)) - ((1 - y_m) .* log(1 - h))));

% Regularized Cost function
T1_rg = Theta1;
T2_rg = Theta2;
% exclude bias
T1_rg(:,1) = 0;
T2_rg(:,1) = 0;
% regularize cost function
rg = (lambda / (2 * m)) * (sum(sum(T1_rg .* T1_rg)) + sum(sum(T2_rg .* T2_rg)));
J = J + rg;

% backpropagation
d3 = (h - y_m);
d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2);

delta1 = (d2' * a1);
delta2 = (d3' * a2);

% regularize gradient
Theta1_grad = (delta1 / m) + (lambda / m) * T1_rg;
Theta2_grad = (delta2 / m) + (lambda / m) * T2_rg;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
