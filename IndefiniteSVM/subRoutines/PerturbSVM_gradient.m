% Implementation of Projected Gradient Method for PerturbSVM
%
% Input:           K: Training Kernel
%              labls: Training labels
%                  c: SVM misclassification penalty parameter
%                rho: Kernel penalty parameter
%           accuracy: precision at which to stop Indefinite SVM
%               info: number of iterations in between gap checks and info printouts
%           maxiters: maximum number of iterations
%                  x: initial point
%             weight: weight for SVM C misclassification parameter
%           stepsize: stepsize for projected gradient method
%
% Output:      alpha: support vectors
%             primal: primal values recorded every "info" iterations
%               dual: dual values recorded every "info" iterations
%                gap: gap recorded every "info" iterations
%            CPUtime: CPUtime recorded every "info" iterations
%          itercount: iteration number recorded every "info" iterations
%
% Ronny Luss and Alexandre d'Aspremont, last modification: March 2008

function [alpha,primal,dual,gap,CPUtime,itercount] = PerturbSVM_gradient(K,labls,c,rho,accuracy,info,maxiters,x,weight,stepsize)
disp('Perturb SVM (using projected gradient) MATLAB starting...');
tic;CPUtime=[];primal=[];dual=[];gap=[];itercount=[]; % initialize output variables
CPUtime=[CPUtime,toc];
n = max(size(K)); % number of training examples
eps=.0001; % parameter for smoothing max function in gradient calculation
P=diag(labls)*K*diag(labls);
libsvmparam= ['-s 0 -c ',num2str(c),' -t 4 -e .0001 -w1 1 -w-1 ',num2str(weight)]; % used to compute dual value
c_weighted=weightPenalty(c,labls,weight); % weight the penalties given the input weight
iter=0;
notconverged=1;
% Begin loop
while(notconverged)
    labls_alpha=labls.*x;
    gradv=ones(n,1)-P*x-(x'*x)*x/(4*rho);% Compute the gradient of f at x

    if mod(iter,info)==0
        primal=[primal;sum(x)-.5*x'*P*x-sum(sum((x*x').^2))/(16*rho)];
        % Compute an upper bound through the dual
        K_opt=K+(labls_alpha*labls_alpha')/(4*rho); % this is the kernel that optimizes the inner problem of the primal
        svmmodel = svmtrain(labls,[(1:length(labls))',K_opt],libsvmparam);
        alpha_temp=zeros(length(labls),1);alpha_temp(svmmodel.SVs)=abs(svmmodel.sv_coef);
        dual=[dual;sum(alpha_temp)-.5*(alpha_temp.*labls)'*K_opt*(alpha_temp.*labls)+rho*trace((K-K_opt)'*(K-K_opt))];
        gap=[gap;dual(end)-primal(end)];
        if (gap(end) < accuracy) % check convergence
            notconverged=0;
        end

        itercount=[itercount;iter];
        CPUtime=[CPUtime,toc];
        disp(['Iter: ',num2str(iter,'%.4e'),'    Primal: ',num2str(primal(end),'%.4e'),'    Dual: ', num2str(dual(end),'%.4e'),'    Gap: ', num2str(gap(end),'%.4e'), '    Time: ', num2str(CPUtime(length(CPUtime)))]);
    end
    if (iter >= maxiters)
        notconverged = 0;
    end;
    if (notconverged)
        x = x + stepsize/(iter+1)*gradv;    % take a step in the direction of the gradient
        x = projcutsimplex4(x,labls,c_weighted,0); % project the new point back to the feasible region
    end
    iter = iter + 1;
end
alpha=x;
