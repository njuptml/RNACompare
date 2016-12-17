% Implementation of Analytic Center Cutting Plane Method for PerturbSVM
%
% Implementation Note: 
% In addition to A_cutplane, we keep track of 2 additional matrices in
% order to eliminate the single equality constraint from SVM constraints:
% AF= A_cutplane*F and AFhat=AF.*AF both for numerical reasons to pass to
% the analytic center subroutine.  F is the matrix we need to multiply by
% to eliminate the equality constraint.

% Input:           K: Training Kernel
%              labls: Training labels
%                  c: SVM misclassification penalty parameter
%                rho: Kernel penalty parameter
%           accuracy: precision at which to stop Indefinite SVM
%               info: number of iterations in between gap checks and info printouts
%           maxiters: maximum number of iterations
%                  x: initial point
%             weight: weight for SVM C misclassification parameter
%
% Output:      alpha: support vectors
%             primal: primal values recorded every "info" iterations
%               dual: dual values recorded every "info" iterations
%                gap: gap recorded every "info" iterations
%            CPUtime: CPUtime recorded every "info" iterations
%          itercount: iteration number recorded every "info" iterations
%
% Ronny Luss and Alexandre d'Aspremont, last modification: December 2008

function [alpha,primal,dual,gap,CPUtime,itercount] = PerturbSVM_ACCPM(K,labls,c,rho,accuracy,info,maxiters,x,weight)
newtonMaxIters=30; % max iterations in newton method for analytic center problem
newtonEpsilon=10e-3; % precision desired in newton method for analytic center problem
tic;CPUtime=[];primal=[];dual=[];gap=[];itercount=[]; % initialize output variables
CPUtime=[CPUtime,toc];
n = max(size(K));
trace_K_K=trace(K'*K);
eps=.0001; % parameter for smoothing max function in gradient calculation
P=diag(labls)*K*diag(labls);
libsvmparam= ['-s 0 -c ',num2str(c),' -t 4 -e .00001 -w1 1 -w-1 ',num2str(weight)]; % used to compute dual value

notconverged=1;
A_cutplane=[];b_cutplane=[];  % stores the gradient inequalities for the analytic centering
AF=[];AFhat=[]; % extra matrices for eliminating equality constraint

c_weighted=weightPenalty(c,labls,weight); % weight the penalties given the input weight
upperbox=c_weighted;
lowerbox=zeros(n,1);

temp=(-1/labls(1))*labls(2:end);
F=[temp';eye(n-1)]; % for eliminating the equality constraint to use Newton's method

iter=0;
% Begin loop
disp('Perturb SVM (using ACCPM) MATLAB starting...');
while(notconverged)
    labls_alpha=labls.*x;
    gradv=ones(n,1)-P*x-(x'*x)*x/(4*rho);% Compute the gradient of f at x
    
    if iter==0
        primal_lb=sum(x)-.5*x'*P*x-sum(sum((x*x').^2))/(16*rho);
    else
        primal_lb=max(primal_lb,sum(x)-.5*x'*P*x-sum(sum((x*x').^2))/(16*rho));
    end
    if mod(iter,info)==0 
        CPUtime=[CPUtime,toc];
        primal_temp=sum(x)-.5*x'*P*x-sum(sum((x*x').^2))/(16*rho);
        primal_lb=max(primal_lb,primal_temp);primal=[primal;primal_lb];

        % Compute an upper bound through the dual
        K_opt=K+(labls_alpha*labls_alpha')/(4*rho); % this is the kernel that optimizes the inner problem of the primal
        svmmodel = svmtrain(labls,[(1:length(labls))',K_opt],libsvmparam);
        alpha_temp=zeros(length(labls),1);alpha_temp(svmmodel.SVs)=abs(svmmodel.sv_coef);
        dual_temp=sum(alpha_temp)-.5*(alpha_temp.*labls)'*K_opt*(alpha_temp.*labls)+rho*trace((K-K_opt)'*(K-K_opt));

        if iter==0
            dual_ub=dual_temp;
        else
            dual_ub=min(dual_ub,dual_temp);
        end
        dual=[dual;dual_ub];
        gap=[gap;dual(end)-primal(end)];

        if (gap(end)) < accuracy % check convergence
            notconverged=0;
        end
        itercount=[itercount;iter];
        disp(['Iter: ',num2str(iter,'%.4e'),'    Primal: ',num2str(primal(end),'%.4e'),'    Dual: ', num2str(dual(end),'%.4e'),'    Gap: ', num2str(gap(end),'%.4e'), '    Time: ', num2str(CPUtime(length(CPUtime)))]);
        
        % Drop some constraints for efficiency
        if iter>0
            [A_cutplane,b_cutplane]=eliminateConstraints(A_cutplane,b_cutplane,AF,F,lowerbox,upperbox,x(2:end));
            AF=A_cutplane*F; % reconstruct AF
            AFhat=AF.*AF;    % reconstruct AFhat
        end
    end
    
    % Compute point for next iterate
    A_cutplane=[A_cutplane;-gradv'];b_cutplane=[b_cutplane;-gradv'*x];
    % we can easily eliminate the single equality constraint: x'*labls=0
    alpha=.01;
    z=x(2:end);
    dirF=(gradv'*F)';
    AF=[AF;-gradv'*F];AFhat=[AFhat;(-gradv'*F).^(2)];
    % line search to find a strictly feasible point
    t=1;count=0;
    while sum([b_cutplane;c_weighted(1);0;-1*lowerbox(2:end);upperbox(2:end)]-[AF;F(1,:);-1*F(1,:);-1*eye(n-1);eye(n-1)]*(z+t*dirF)<=0)>0
        t=alpha*t;
        if count > 10 % eliminiate constraints causing 'numerical difficulties'
            ind =find(b_cutplane-AF*(z+t*dirF)>0);
            A_cutplane=A_cutplane(ind,:);
            AF=AF(ind,:);
            AFhat=AFhat(ind,:);
            b_cutplane=b_cutplane(ind);
            t=1;
            count=0;
        end
        count=count+1;
    end % end of line search

    z=z+t*dirF;  % take a step so the initial point is strictly feasible
    z=AnalyticCenterCG_mex([AF;F(1,:);-1*F(1,:)],[b_cutplane;c_weighted(1);0],lowerbox(2:end),upperbox(2:end),newtonMaxIters,newtonEpsilon,z,[AFhat;F(1,:).^2;F(1,:).^2]);
    x=F*z;
    if (iter >= maxiters)
        notconverged = 0;
    end;
    iter = iter + 1;
end;
alpha=x;
