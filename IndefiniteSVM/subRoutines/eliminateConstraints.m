% Eliminate constraints in ACCPM
%
% Ronny Luss and Alexandre d'Aspremont, last modification: December 2008

function [A_cutplane,b_cutplane] = eliminateConstraints(A_cutplane,b_cutplane,AF,F,lowerbox,upperbox,z)
n=length(z);

% compute inv(H)
if isempty(A_cutplane)==0 
    H=[AF;F(1,:);-1*F(1,:)]./(([b_cutplane;upperbox(1);lowerbox(1)]-[AF;F(1,:);-1*F(1,:)]*z)*ones(1,n));
    H=H'*H+diag((upperbox(2:end)-z).^(-2)+(z-lowerbox(2:end)).^(-2));
    Hinv=pinv(H); % dirty for bad conditioning
else 
    H=(upperbox(2:end)-z).^(-2)+(z-lowerbox(2:end)).^(-2); 
    Hinv=diag(H.^(-1));
end

relevance=(b_cutplane-AF*z)./(diag(AF*Hinv*AF').^.5); % defined measure of relevance
keep=(relevance<size(A_cutplane,1));
A_cutplane=A_cutplane(keep,:); % first drop constraints that are redundant
b_cutplane=b_cutplane(keep,:);
relevance=relevance(keep);
if length(relevance)>3*n % then drop constraints if we have too many
    [temp,ind]=sort(relevance,'ascend');
    A_cutplane=A_cutplane(ind(1:3*n),:);
    b_cutplane=b_cutplane(ind(1:3*n));
end


    
  
