% Implementation of Projected Gradient Method for IndefiniteSVM
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

function [alpha,primal,dual,gap,CPUtime,itercount] = IndefiniteSVM_gradient(K,labls,c,rho,accuracy,info,maxiters,x,weight,stepsize)
disp('Indefinite SVM (using projected gradient) MATLAB starting...');
tic;CPUtime=[];primal=[];dual=[];gap=[];itercount=[]; % initialize output variables
CPUtime=[CPUtime,toc];
n = max(size(K)); % number of training examples
eps=.0001; % parameter for smoothing max function in gradient calculation
% compute eigenvalue decomposition of input kernel to use for eigenvalue
% decomposition of rank-one updates
[V0,D0]=eig(K); 
D0=diag(D0);
trace_K_K=D0'*D0; % constant in the primal objective value
libsvmparam= ['-s 0 -c ',num2str(c),' -t 4 -e .0001 -w1 1 -w-1 ',num2str(weight)]; % used to compute dual value

c_weighted=weightPenalty(c,labls,weight); % weight the penalties given the input weight
iter=0; % iteration count
notconverged=1;
while(notconverged)
    labls_alpha=labls.*x;
    % Compute the gradient of f at x
    [V,D]=eigUpdateMult_mex(V0,D0,V0'*labls_alpha,1/(4*rho));
    if sum(sum(isnan([V,D])))>0 % for numerical stability
        [V,D]=eig(K+labls_alpha*labls_alpha'/(4*rho));D=diag(D);
%         disp(['eigUpateMult_mex failed at iteration ',num2str(iter),' with C=',num2str(c)]);
    end

    % labls_alpha'*V is sum(l=1..n) y_l*alpha_l*v_{i,l} where v_{i,l} is the lth component of the ith eigenvector
    gradf=V.*(labls*(labls_alpha'*V))/(2*rho); % gradf(i,j) is now the derivative of the ith eigenvalue wrt to alpha(j)
    
    % compute derivative of smoothed max function
    u=(D/eps);u=zeros(n,1)+(u>1)+u.*(u<=1).*(u>=0);
    gradf=gradf.*(u*ones(1,n)); % gradf(i,j) is now the derivative of the smoothed max(0,ith eigenvalue) wrt to alpha(j)
    
    % compute gradient of smoothed objective
    grad_traces=D-((V'*labls_alpha).^2)/(4*rho); % equivalent to diag(V'*K*V)
    gradv=ones(n,1)+(-.5*(((labls_alpha)'*V).^2)*gradf)';
    gradv=gradv-(labls.*(V*(D.*(D>0).*((labls_alpha)'*V)')));
    gradv=gradv+2*rho*gradf'*(D.*(D>0));
    gradv=gradv-2*rho*gradf'*grad_traces;

    if mod(iter,info)==0    
        CPUtime=[CPUtime,toc];
        D=D.*(D>0);primal=[primal;sum(x)-.5*((labls_alpha'*V).^2)*D+rho*D'*D-2*rho*grad_traces'*D+rho*trace_K_K];
        % Compute an upper bound to the dual obejctive value
        K_opt=V*diag(D)*V'; % K_opt optimizes the inner minimization of the primal problem
        svmmodel = svmtrain(labls,[(1:length(labls))',K_opt],libsvmparam);
        alpha_temp=zeros(length(labls),1);alpha_temp(svmmodel.SVs)=abs(svmmodel.sv_coef);
        dual=[dual;sum(alpha_temp)-.5*(alpha_temp.*labls)'*K_opt*(alpha_temp.*labls)+rho*trace((K_opt-K)'*(K_opt-K))];

        gap=[gap;dual(end)-primal(end)];
        if (gap(end) < accuracy) % check convergence
            notconverged=0;
        end
        itercount=[itercount;iter];

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
end;
alpha=x;
