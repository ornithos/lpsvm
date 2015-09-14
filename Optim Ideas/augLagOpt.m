function x = augLagOpt(f, A, b, x0, maxIter, tol, eps)
% augLagOpt Augmented Lagrangian Optimisation of the LPSVM Dual Problem
%   EVTUSHENKO ET AL.
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
[m, n] = size(A);
beta = 0.1;
if length(maxIter) == 1
    maxIter = repmat(maxIter,2,1);
end

currX = x0;
p = ones(m,1);

for k = 1:maxIter(1)
    
    % Solution of maximisation problem
    for k = 1:maxIter(2)
        z = currX + A'*p - beta.*f;
        S = b - A*vecPos(z);
        indD = diag(z>eps);
        genHess = A*indD*A';

        % regularisation/modification to ensure nonsingular hessian
        delta = 1e-5;
        step = (genHess + delta*eye(m))\S;
        lsP = @(l) p - l.*step;
        lsFn = @(l) b'*lsP(l) - 0.5*euclNormsq(vecPos(currX + A'*lsP(l) - beta.*f));
        lambda = fminbnd(lsFn, 0, 1);
        
        % Update p and check convergence
        pOld = p;
        p = p - lambda.*step;
        if abs(pOld - p) < tol; break; end;
    end

    currX = vecPos(currX + A'*p - beta.*f);
    u = p/beta;
    if max(A*currX-b) <= tol && ...
            max(vecPos(A'*u -c)) <= tol && ...
            abs(f'*currX-b'*u) <= tol
        break
    end
end

x = currX;
end

function v = vecPos(x)
    v = (abs(x)+x)/2;
end

function y = euclNormsq(x)
    y = x'*x;
end