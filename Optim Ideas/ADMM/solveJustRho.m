function rho = solveJustRho(s, D)
%solveJustRho(s) 
% For a given value of a, we can solve for rho relatively easily. This
% function returns the optimal rho for (rho-s)_(+) - rho
n = length(s);
s = sort(s, 'ascend');
cs = [0; cumsum(s(1:end-1))];
x = ((0:(n-1))-1/D)'.*s - cs;
[~,mn] = min(x);
rho = s(mn);
end

