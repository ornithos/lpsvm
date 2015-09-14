function [a, rho, xi, fval] = solve2DBisection(H, D)
% solve2DBisection(H)
%   H is a 2 row matrix. We minimise the function
%   rhoObj = @(x,j) sum(max(0,x-H'*[j; 1-j])) - x;

[n,~] = size(H);
if n ~=2; error('H must be a 2 column matrix'); end;

tol = 1e-10;   % all evaluations are basically free, so 10^-10 seems good.
opts = optimset('TolX',tol,'MaxIter',200,'Display','None');
j = fminbnd(@(j)solveGivenJ(H,j,D),0, 1, opts);

a = [j; 1-j];
s = H'*a;
rho = solveJustRho(s, D);
xi = max(rho-s,0);
fval = D*sum(max(0,rho-s)) - rho;
end



function out = solveGivenJ(H, j, D)
s = H'*[j; 1-j];
rho = solveJustRho(s, D);
out = D*sum(max(0,rho-s)) - rho;
end