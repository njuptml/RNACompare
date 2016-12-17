% estimate the b parameter to the SVM decision function
%
% Input:         alpha: alpha outputfrom SVM
%               Ktrain: train kernel input to SVM
%               labels: labels for training instances
%                    C: C SVM parameter 
%               weight: weight for SVM C misclassification parameter
% 
% Output:            b: estimated b parameter
%
% Ronny Luss and Alexandre d'Aspremont, last modification: March 2008

function b = estimate_b(alpha,Ktrain,labels,C)

thresh=1e-3;
weight=sum(labels>0)/sum(labels<0);
C_weighted=weightPenalty(C,labels,weight); % weight the penalties given the input weight
num=sum((alpha>thresh).*(alpha<C_weighted-thresh)); % number of alpha's not at bounds
if num>0 % i.e. there are support vectors
    b=sum((labels-Ktrain*(alpha.*labels)).*(alpha>thresh).*(alpha<C_weighted-thresh))/num;
else 
    b=0;
end
