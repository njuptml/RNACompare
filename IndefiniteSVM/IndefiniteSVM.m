% Wrapper for Indefinite SVM
% 
% Input:           K: Training Kernel
%             labels: Training labels
%                  C: SVM misclassification penalty parameter
%                rho: Kernel penalty parameter
%           accuracy: precision at which to stop Indefinite SVM
%               info: number of iterations in between display statements (and gap checks)
%           maxiters: maximum number of iterations
%           stepsize: stepsize for projected gradient algorithm if type=1
%               type: 1 for Projected Gradient Algorithm, 2 for ACCPM
%
% Output:      alpha: support vectors
%             primal: primal values recorded every "info" iterations
%               dual: dual values recorded every "info" iterations
%                gap: gap recorded every "info" iterations
%            CPUtime: CPUtime recorded every "info" iterations
%          itercount: iteration number recorded every "info" iterations
%
% Ronny Luss and Alexandre d'Aspremont, last modification: March 2008

function [alpha,primal,dual,gap,CPUtime,itercount] = IndefiniteSVM(K,labels,C,rho,accuracy,info,maxiters,stepsize,type)
weight=sum(labels>0)/sum(labels<0); % weight the SVM C penalties for over/under representation
C_weighted=weightPenalty(C,labels,weight);
x0=projcutsimplex4(C_weighted/2,labels,C_weighted,0); % compute an initial point

switch type
    case{1} % Projected Gradient Algorithm
        [alpha,primal,dual,gap,CPUtime,itercount]=IndefiniteSVM_gradient(K,labels,C,rho,accuracy,info,maxiters,x0,weight,stepsize);

    case{2} % Analytic Center Cutting Plane Algorithm
        [alpha,primal,dual,gap,CPUtime,itercount]=IndefiniteSVM_ACCPM(K,labels,C,rho,accuracy,info,maxiters,x0,weight);
end
