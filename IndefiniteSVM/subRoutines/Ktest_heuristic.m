% Compute the test kernel for Indefinite SVM using a heuristic
% Input:            K: input kernel
%               alpha: alpha results from SVM on training kernel
%         labelsTrain: training labels needed for the heuristic
%                 rho: penalty parameter to Indefinite SVM
% 
% Output:       Ktest: testing kernel (m x l matrix for m test instances and l
%                      training instances)
%
% Ronny Luss and Alexandre d'Aspremont, last modification: March 2008

function Ktest = Ktest_heuristic(K,alpha,labelsTrain,rho)

numtrain=length(labelsTrain);
offset=zeros(size(K));
offset(1:numtrain,1:numtrain)=(labelsTrain.*alpha)*(labelsTrain.*alpha)'/(4*rho);
[V,D]=eig(K+offset); % decompose K plus the rank-one heuristic for the offset
Ktest=V*diag(diag(D).*(diag(D)>0))*V'; % project onto PSD matrices
Ktest=Ktest((numtrain+1):end,1:numtrain); % only return test elements of kernel
