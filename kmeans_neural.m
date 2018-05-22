%     min - Tr X'XU'U     U is Ytilde
%     U'U1 = 1  - leads to z<>0
%     U >= 0
%  Reduces to LP,solved by primal-dual
%
%  based on kmeansLP_seqActiv
%
function [Y,w,Z,whistory] = kmeansLP_init( X, w,...
    neural... %if neural=='neural', use neural plausible primal/dual, othw LP
    )
if nargin>=3 && strcmp(neural,'neural'), LP=false; 
else LP=true; end
%%
[n,T]=size(X);
k=size(w,2); %active dims
whistory=w(:);
ycum=ones(k,1);  %  !! IMPORTANT for shift-invariance; was eps*ones(k,1);
Y=zeros(k,T);
Z=zeros(1,T);
z=0;
v=zeros(k,1);

%%
eta=.01; %'neural' only
rho=.5; %'neural' only
y=zeros(k,1); %'neural' only
for t=1:T
    x = X(:,t);
    a = 2*w'*x - v;

    if LP % non-neural
        [z,iwinner] = max(a);
        y=zeros(k,1); y(iwinner)=1;

    else    % Primal-Dual Subgrad Descent
        er = 1;
        iter=1;
        while er > 1e-7

             Z_prev = z;
             Y_prev = y;

             y = max( 0, Y_prev - (eta)*( -a + z + rho*(sum(y) - 1) ) );
             z = Z_prev + eta*( sum(y) - 1 );    
             %z = max( 0, Z_prev + eta*( sum(y) - 1 ) );    

             er = max(norm(Z_prev-z)./[norm(Z_prev)+1e-4],norm(Y_prev-y)./[norm(Y_prev)+1e-4]);
             iter=iter+1;

        end
    end

    ycum = ycum + y;
    w = w + (X(:,t)*ones(1,k) - w) * diag(y./ycum);
    v = v + diag(y./ycum) * (z - v);
    Y(:,t) = y;
    Z(t) = z;
    if nargout>=4
        whistory(:,t+1)=w(:);  %whistory=[whistory, w(:)]
    end
end
