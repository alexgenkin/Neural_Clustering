% rand index btw two classifications
% 
%  #pairs-together-in-both + #pairs-separate-in-both
%  -------------------------------------------------
%         all-pairs
%         
        
function r = randIndex(c1,c2) 

C = crosstab(c1,c2);
n = sum(C(:));
allpairs = n*(n+1)/2;
%together = sum(sum( C.*(C+1)/2 ));
% separate in c1, together in c2
sep1 = ( sum( sum(C,2).^2 )  - sum(sum(C.^2)) )/2;
% separate in c2, together in c1
sep2 = ( sum( sum(C,1).^2 )  - sum(sum(C.^2)) )/2;

r = (allpairs - sep1 - sep2)/allpairs; 
