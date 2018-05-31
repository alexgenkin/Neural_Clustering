%
%  original k-means, no bio plausible effort
% v2 fix: sum_y=zeros
%
function [Y,w,whistory] = kmeans_online( X, winit )
[n,T]=size(X);  %input dim
k=size(winit,2);  % output dim = #clusters
w=winit;%+0.2*randn(2,k);
whistory=w(:);
sum_y=zeros(k,1);
Y=zeros(k,T);
%% 
for t=1:T
    x = X(:,t);
    d2 = sum((x*ones(1,k) - w).^2)'; % squared distances
    [~,imin] = min(d2);
    y=zeros(k,1);
    y(imin) = 1;
    Y(:,t) = y;
    sum_y(imin) = sum_y(imin) + 1;
    w(:,imin) = w(:,imin) + (x - w(:,imin)) / sum_y(imin);
    if nargout>=3
        whistory=[whistory, w(:)]; end
end
