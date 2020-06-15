function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)

    predictions = (pval < epsilon);
    % True Positive, prediction is possitive and groud truth is positive
    tp = sum(predictions == 1 & yval == 1);
    % False Positive, prediction is possitive but groud truth is negative
    fp = sum(predictions == 1 & yval == 0);
    % False Negative, prediction is negative but groud truth is positive
    fn = sum(predictions == 0 & yval == 1);

    % precision and recall
    if ((tp+fp == 0) || (tp+fn == 0))
      continue
    end
    prec = tp / (tp + fp);
    rec = tp / (tp + fn);

    % F1 score
    F1 = (2 * prec * rec) / (prec + rec);

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
