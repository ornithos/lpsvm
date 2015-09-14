function [a, rho, xi, fval] = solve2DBisection2(H, D, rng_start, rng_end)
% solve2DBisection2(H, D, start, end)
%   for general H. We minimise the function
%   rhoObj = @(x,a) sum(max(0,x-H'*a)) - x;
%   where a is on the line [rng_start, rng_end].

tol = 1e-10;   % all evaluations are basically free, so 10^-10 seems good.
opts = optimset('TolX',tol,'MaxIter',200,'Display','None');

dir = rng_end - rng_start;

j = fminbnd(@(j)solveGivenJ(H,rng_start,dir,j,D),0, 1, opts);

a = rng_start + j*dir;
s = H'*a;
rho = solveJustRho(s, D);
xi = max(rho-s,0);
fval = D*sum(max(0,rho-s)) - rho;
end



function out = solveGivenJ(H, start, dir, j, D)
a = start + j*dir;
s = H'*a;
rho = solveJustRho(s, D);
out = D*sum(max(0,rho-s)) - rho;
end