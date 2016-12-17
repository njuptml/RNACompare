% compute accuracy, recall, and precision
%
% Ronny Luss and Alexandre d'Aspremont, last modification: March 2008

function [accuracy,recall,precision] = calcMeasures(pred,labels)

indpos=find(pred==1);indneg=find(pred==-1);
truepos=sum(pred(indpos)==labels(indpos));
trueneg=sum(pred(indneg)==labels(indneg));
falsepos=sum(pred(indpos)~=labels(indpos));
falseneg=sum(pred(indneg)~=labels(indneg));

accuracy=sum(pred==labels)/length(labels);
if accuracy<.001
    ronny=3;
end
precision=trueneg/(trueneg+falsepos);
recall=truepos/(truepos+falseneg);
