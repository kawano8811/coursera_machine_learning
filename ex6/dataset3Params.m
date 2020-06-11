function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;


C_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

err_size = size(C_list)(2) * size(sigma_list)(2);
results = [zeros(err_size, 3)];
idx = 1;
for i=[1:size(C_list)(2)]
  for j=[1:size(sigma_list)(2)]
    model= svmTrain(X, y, C_list(i), @(x1, x2) gaussianKernel(x1, x2, sigma_list(j))); 
    predictions = svmPredict(model, Xval);
    results(idx, 1) = mean(double(predictions ~= yval));
    results(idx, 2) = C_list(i);
    results(idx, 3) = C_list(j);
    idx = idx + 1;
  endfor
endfor

[err, row] = min(results(:, 1));
C = results(row, 2);
sigma = results(row, 3);

end
