% Implementation of Analytic Center Cutting Plane Method for Indefinite1classSVM
%
% Implementation Note: 
% In addition to A_cutplane, we keep track of 2 additional matrices in
% order to eliminate the single equality constraint from SVM constraints:
% AF= A_cutplane*F and AFhat=AF.*AF both for numerical reasons to pass to
% the analytic center subroutine.  F is the matrix we need to multiply by
% to eliminate the equality constraint.

% Input:           K: Training Kernel
%              labls: Training labels
%                  nu: one-class SVM parameter
%                rho: Kernel penalty parameter
%           accuracy: precision at which to stop Indefinite SVM
%               info: number of iterations in between gap checks and info printouts
%           maxiters: maximum number of iterations
%                  x: initial point
%
% Output:      alpha: support vectors
%             primal: primal values recorded every "info" iterations
%               dual: dual values recorded every "info" iterations
%                gap: gap recorded every "info" iterations
%            CPUtime: CPUtime recorded every "info" iterations
%          itercount: iteration number recorded every "info" iterations
%
% Ronny Luss and Alexandre d'Aspremont, last modification: December 2008

function [alpha,primal,dual,gap,CPUtime,itercount] = Indefinite1classSVM_ACCPM(K,nu,rho,accuracy,info,maxiters,x)
newtonMaxIters=30; % max iterations in newton method for analytic center problem
newtonEpsilon=10e-3; % precision desired in newton method for analytic center problem
tic;CPUtime=[];primal=[];dual=[];gap=[];itercount=[]; % initialize output variables
CPUtime=[CPUtime,toc];
n = max(size(K));
trace_K_K=trace(K'*K);
eps=.0001; % parameter for smoothing max function in gradient calculation
iter=0; % iteration count
labls=ones(n,1); % proxy labels for the single class

% compute eigenvalue decomposition of input kernel to use for eigenvalue decomposition of rank-one updates
[V0,D0]=eig(K); 
D0=diag(D0);
libsvmparam= ['-s 2 -t 4 -n ',num2str(nu),' -e .0001'];

A_cutplane=[];b_cutplane=[];  % stores the gradient inequalities for the analytic centering
AF=[];AFhat=[]; % extra matrices for eliminating equality constraint

upperbox=ones(n,1)/nu/n;
lowerbox=zeros(n,1);

temp=-ones(n-1,1);
F=[temp';eye(n-1)]; % for eliminating the equality constraint to use Newton's method
xhat=[1;zeros(n-1,1)]; % transformation is x=Fz+xhat

iter=0;
notconverged=1;
% Begin loop
disp('Indefinite one-class SVM (using ACCPM) MATLAB starting...');
while(notconverged)
    % Compute the gradient of f at x
    [V,D]=eigUpdateMult_mex(V0,D0,V0'*x,1/(4*rho));
    if sum(sum(isnan([V,D])))>0 % for numerical stability
        [V,D]=eig(K+x*x'/(4*rho));D=diag(D);
%         disp(['eigUpateMult_mex failed at iteration ',num2str(iter),' with C=',num2str(c)]);
    end

    % x'*V is sum(l=1..n) x_l*v_{i,l} where v_{i,l} is the lth component of the ith eigenvector
    gradf=V.*(ones(n,1)*(x'*V))/(2*rho); % gradf(i,j) is now the derivative of the ith eigenvalue wrt to alpha(j)
    
    % compute derivative of smoothed max function
    u=(D/eps);u=zeros(n,1)+(u>1)+u.*(u<=1).*(u>=0);
    gradf=gradf.*(u*ones(1,n)); % gradf(i,j) is now the derivative of the smoothed max(0,ith eigenvalue) wrt to alpha(j)
        
    % compute gradient of smoothed objective
    grad_traces=D-((V'*x).^2)/(4*rho); % equivalent to diag(V'*K*V)
    gradv=(-.5*((x'*V).^2)*gradf)';
    gradv=gradv-(ones(n,1).*(V*(D.*(D>0).*(x'*V)')));
    gradv=gradv+2*rho*gradf'*(D.*(D>0));
    gradv=gradv-2*rho*gradf'*grad_traces;
    
    D=D.*(D>0);
    if iter==0
        primal_lb=-.5*((x'*V).^2)*D+rho*D'*D-2*rho*grad_traces'*D+rho*trace_K_K;
    else
        primal_lb=max(primal_lb,-.5*((x'*V).^2)*D+rho*D'*D-2*rho*grad_traces'*D+rho*trace_K_K);
    end
    if mod(iter,info)==0 
        CPUtime=[CPUtime,toc];
        primal_temp=-.5*((x'*V).^2)*D+rho*D'*D-2*rho*grad_traces'*D+rho*trace_K_K;
        primal_lb=max(primal_lb,primal_temp);primal=[primal;primal_lb];

        % Compute an upper bound through the dual
        K_opt=V*diag(D)*V'; % this is the kernel that optimizes the inner problem of the primal
        svmmodel = svmtrain(labls,[(1:length(labls))',K_opt],libsvmparam);
        alpha_temp=zeros(length(labls),1);alpha_temp(svmmodel.SVs)=svmmodel.sv_coef;
        alpha_temp=alpha_temp/nu/n;  % scale back because LIBSVM solves scaled version of one-class SVM        
        dual_temp=-.5*alpha_temp'*K_opt*alpha_temp+rho*trace((K_opt-K)'*(K_opt-K));

        if iter==0, dual_ub=dual_temp; else dual_ub=min(dual_ub,dual_temp); end
        dual=[dual;dual_ub];
        gap=[gap;dual(end)-primal(end)];

        if (gap(end)) < accuracy % check convergence
            notconverged=0;
        end
        itercount=[itercount;iter];
        disp(['Iter: ',num2str(iter,'%.4e'),'    Primal: ',num2str(primal(end),'%.4e'),'    Dual: ', num2str(dual(end),'%.4e'),'    Gap: ', num2str(gap(end),'%.4e'), '    Time: ', num2str(CPUtime(length(CPUtime)))]);
        
        % Drop some constraints for efficiency
%         if iter>0
%             [A_cutplane,b_cutplane]=eliminateConstraints(A_cutplane,b_cutplane,AF,F,lowerbox-xhat,upperbox-xhat,x(2:end));
%             AF=A_cutplane*F; % reconstruct AF
%             AFhat=AF.*AF;    % reconstruct AFhat
%         end
    end
    
    % Compute point for next iterate
    A_cutplane=[A_cutplane;-gradv'];b_cutplane=[b_cutplane;-gradv'*x+gradv'*xhat];
    % we can easily eliminate the single equality constraint: x'*labls=0
    alpha=.01;
    z=x(2:end);
    dirF=(gradv'*F)';
    AF=[AF;-gradv'*F];AFhat=[AFhat;(-gradv'*F).^(2)];
    % line search to find a strictly feasible point
    t=1;count=0;
    while sum([b_cutplane;1/nu/n-1;1;-1*lowerbox(2:end);upperbox(2:end)]-[AF;F(1,:);-1*F(1,:);-1*eye(n-1);eye(n-1)]*(z+t*dirF)<=0)>0
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
    z=AnalyticCenterCG_mex([AF;F(1,:);-1*F(1,:)],[b_cutplane;1/nu/n-1;1],lowerbox(2:end),upperbox(2:end),newtonMaxIters,newtonEpsilon,z,[AFhat;F(1,:).^2;F(1,:).^2]);
    x=F*z+xhat;
    if (iter >= maxiters)
        notconverged = 0;
    end;
    iter = iter + 1;
end;
alpha=x;
