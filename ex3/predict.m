function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

X = [ones(m, 1) X]; % add bias
a2 = sigmoid(X * Theta1');
a2 = [ones(m, 1) a2]; % add bias
a3 = sigmoid(a2 * Theta2');

[x, ix] = max(a3, [], 2);
p = ix; % max index







% =========================================================================


end
