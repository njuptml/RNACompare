% return a vector of penalties where Cweighted(i)=C if labels(i)=1
% and Cweighted(i)=weight*C if labels(i)=-1
%
% Ronny Luss and Alexandre d'Aspremont, last modification: March 2008

function Cweighted = weightPenalty(C,labels,weight)

Cweighted=C*ones(length(labels),1);
Cweighted(labels<0)=Cweighted(labels<0)*weight;
