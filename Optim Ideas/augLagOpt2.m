function [x, y] = augLagOpt2(f, A, b, x0, maxIter, tol, lambda)
% augLagOpt 2nd Augmented Lagrangian Optimisation of the LPSVM Dual Problem
%   GULER
%   This is similar to a Method of Multipliers approach

%   Arguments:
%   f          -   (Col Vector) Objective function
%   A          -   (Matrix) equality constraint for Ax = b
%   b          -   (Col Vector) equality constraint for Ax = b
%   x0         -   Initial (warm start) x
%   maxIter    -   maximum number of iterations in FW loop
%   tol        -   stopping criterion.
%   eps        -   tolerance for determining equality

% IGNORING SPARSE LINEAR ALGEBRA FOR NOW
if size(f,2)>1; f = f'; end;
if size(b,2)>1; b = b'; end;
if size(x0,2)>1; x0 = x0'; end;

[m, n] = size(A);
lb = zeros(n,1);
qpOpts = optimoptions(@quadprog, 'Display', 'None');

if isempty(x0)
    x0 = zeros(n,1);
end
currX = x0;
currY = zeros(m,1);
currY = yUpdate(currY, currX, A, b, lambda);

%    Warm Start
%    Choice of lambda 
%    SPARSE ALGEBRA

for k = 1:maxIter
    [currX, ~, exitflag] = quadprog(lambda.*A'*A, f - A'*currY - lambda.*A'*b, ...
        [], [], [], [], lb, [], [], qpOpts);
    
    currY = yUpdate(currY, currX, A, b, 1);
    
    dualityGap = f'*currX-b'*currY;
    convg = [max(abs(A*currX-b)), max(A'*currY - f), abs(dualityGap)];
    if all(convg < n*tol); break; end;
end

if ~all(convg < n*tol)
    fprintf('\n (augLagOpt2) Optimum not reached within %d iters\n', maxIter);
else
    fprintf('\n (augLagOpt2) Converged in %d iters\n', k);
end

x = currX; y = currY;
end

function y = yUpdate(yOld, x, A, b, lambda)
    y = yOld + lambda*(b-A*x);
end