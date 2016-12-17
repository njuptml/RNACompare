% calculate SVM error for Indefinite SVM
%
%  Input:       alpha: alpha outputfrom SVM
%                   b: b output from SVM
%                   K: full kernel containing training and test data
%         labelsTrain: labels for training instances
%          labelsTest: labels for test instances
%                           
% Output:        pred: predicted values
%            accuracy: (TP+TN)/(TP+FP+TN+FN)
%              recall: TP/(TP+FN)
%           precision: TP/(TP+FP)
%
% Ronny Luss and Alexandre d'Aspremont, last modification: March 2008

function [pred,accuracy,recall,precision] = IndSVMerror(alpha,labelsTrain,labelsTest,K,C,rho)

numTrain=length(labelsTrain);
b=estimate_b(alpha,K(1:numTrain,1:numTrain),labelsTrain,C); % estimate b parameter
Ktest=Ktest_heuristic(K,alpha,labelsTrain,rho); % get the test kernel
pred=sign(Ktest*(alpha.*labelsTrain)+b); % make the predictions
[accuracy,recall,precision] = calcMeasures(pred,labelsTest); % calculate measures
disp([' Accuracy: ', num2str(accuracy,'%.4f')]);
disp(['   Recall: ', num2str(recall,'%.4f')]);
disp(['Precision: ', num2str(precision,'%.4f')]);

                               