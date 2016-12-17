% Ronny Luss and Alexandre d'Aspremont, last modification: March 2008

function [resb,lambda]=projcutsimplex4(x,a,c,t)
% Computes the Euclidean projection of x on the set 
% 0 <= x_i <= c_i ,  a'*x = t

% Prune out indices for which a_i=0
za=find(abs(a)<1e-10);nza=find(abs(a)>1e-10);
resb(za)=min([c(za)';max([zeros(size(c(za)))';x(za)'])])';
x=x(nza);a=a(nza);c=c(nza);

n=size(x,1);
e=ones(size(x));

anf=[a;+inf];
[veca,inda]=sort([-2*x./a;+inf]);
[vecb,indb]=sort([2*(c-x)./a;+inf]);
asa=anf(inda);asb=anf(indb);

[lastpoint,type]=min([veca(1),vecb(1)]);
grad=t-c(find(a<0))'*a(find(a<0));
gslope=0;
if type==1
    ai=asa(1);
    veca(1)=[];asa(1)=[];
    if ai>=0
        gslope=gslope-ai^2/2;
    else
        gslope=gslope+ai^2/2;
    end
else
    ai=asb(1);
    vecb(1)=[];asb(1)=[];
    if ai>=0
        gslope=gslope+ai^2/2;
    else
        gslope=gslope-ai^2/2;
    end
end


while min([veca(1),vecb(1)])<inf
    [point,type]=min([veca(1),vecb(1)]);
    interval=point-lastpoint;lastpoint=point;
    grad=grad+interval*gslope;
    if grad<0 break; end;
    if type==1
        ai=asa(1);
        veca(1)=[];asa(1)=[];
        if ai>=0
            gslope=gslope-ai^2/2;
        else
            gslope=gslope+ai^2/2;
        end
    else
        ai=asb(1);
        vecb(1)=[];asb(1)=[];
         if ai>=0
            gslope=gslope+ai^2/2;
        else
            gslope=gslope-ai^2/2;
        end
    end;
end
lambda=point-grad/gslope;
res=e;
for i=1:n
    res(i)=x(i)+(lambda*a(i))/2;
    if res(i)<0 res(i)=0; end;
    if res(i)>c(i) res(i)=c(i); end;
end
resb(nza)=res;resb=resb';