% no update, use fixed centroids w, sum_y, sum_zy
%  neural: uses z interneuron, but inner loop neural is optional
%
%  min - Tr X'XU'U  + alpha*T Tr U'U    - U is Ytilde
%     U'U1 = 1                       - i.e. z pos or neg
%     U >= 0
% 
% v.3 alpha*T, but no x quad term; derived from kmeans_L2_seqActiv.m
% v.4 precise algorithm instead of quadprog, mod of Wang, Carreira-Perpi?n´an 2013, arxiv 1309.1541
%
function [Y] = kmeans_L2_neural_noupd( X, alpha, w, sum_y, sum_zy,...
    neural,... % use neural plausible primal/dual if =='neural'
    eta, rho ) % primal/dual params
if nargin>=6 && strcmp(neural,'neural'), precise=false;
else, precise=true;end
if nargin<7, eta=.01; end
if nargin<8, rho=1; end
%%
[n,T]=size(X);  % input dim
k=size(w,2);  % output dim
Y=zeros(k,T);
Z=zeros(1,T);
z=0;y=zeros(k,1);
%% 
tol=1e-8;
for t=1:T
    x = X(:,t);
    a = 2*w'*x - sum_zy;
    % min -a'*y + alpha*t y'./diag(sum_y)*y
    %  s.t. y'*1=1, y>=0
    if precise %     non-neural
        if alpha==0
            [z,iwinner] = max(a);
            y=zeros(k,1); y(iwinner)=1;
        else
            d = 2*alpha*t*(1./sum_y);
            [asort,aord] = sort(a,'descend');
            dsort = d(aord);
            ztry = (cumsum(asort./dsort) - 1) ./ cumsum(1./dsort);
            %- find best
            bestObj=Inf; bestP=0;
            for p=1:k
                if find(asort>ztry(p),1,'last')==p
                    y = max(0, (a-ztry(p))./d);
                    objval = -a'*y + (y.^2)'*d/2;  %-sum( (asort(1:p).^2 + ztry(p)^2) ./ dsort(1:p)/2 );
                    if objval<bestObj
                        bestObj = objval;
                        bestP = p;
                    end 
                end
            end
            z = ztry(bestP);
            y = max(0, (a-z)./d);
        end
    else % neural plausible primal/dual
        er = 1;
        iter=1;
        z=0;
        while er > tol

             Z_prev = z;
             Y_prev = y;

             above = max(0,sum(Y_prev) - 1);
             y = max( 0, Y_prev - (eta)*( -a + 2*(alpha*t)*Y_prev./sum_y + ...
                                        (Z_prev + rho*above)... should be *dabove, but that's not working 
                                        ) );
             z = Z_prev + eta*( above ); %was for z>=0: max( 0, );    

             er = max(norm(Z_prev-z)./[norm(Z_prev)+1e-4],norm(Y_prev-y)./[norm(Y_prev)+1e-4]);
             iter=iter+1;

        end
    end
    
    Y(:,t) = y;
    Z(t) = z;
end
%figure,plot(zhistory');