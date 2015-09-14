function [u, beta, exitflag] = frankWolfe2(H, initial, D, maxIter, epsilon, ...
    tol, searchType, dbg)
% Second Frank-Wolfe (Reverse) solution of the minimax problem for LPSVM.
% Here we have relaxed the problem to a continuous set consisting of all
% convex combinations of H, and used Von Neumann/Sion Theorem to view the
% problem in reverse. As it (should) turn out, the relaxation is w.l.o.g.
%
%   Will output both the optimal u and the minimax achieved, beta.
%
%   Arguments:
%   H          -   (Matrix) rows = hypotheses / cols = datapoints
%   initial    -   initial (feasible) point. Can be used for warm start.
%   D          -   largest permitted value for box constraints
%   maxIter    -   maximum number of iterations in FW loop
%   epsilon    -   stopping criterion.
%   tol        -   tolerance for determining if two H_j u are equal (in max)
%   searchType -   for gamma. 0 = 2/(k+2), 1 = line search.
%   dbg        -   verbose option. Not many extra messages at present.
%
%   Currently using D White generalisation for Frank-Wolfe to non
%   differentiable case. Still not working well.
%
%   IMPORTANT NOTE: SPARSITY HAS NOT YET BEEN EXPLOITED WITHIN THE
%   ALGORITHM. SPEEDUPS WILL BE POSSIBLE BY NOT EVALUATING INNER PRODUCTS
%   FOR ALL u_i = 0.

% Setup: Admin
[p, N] = size(H);
if ~all(size(initial) == [1, p])
    error('(Frank-Wolfe) initial vector is of size [%d,%d]. Should be [1,%d].\n', ...
        size(initial,1), size(initial,2), p);
end
currX = initial;

% Setup: Save intermediate steps for debug purposes.
saveSkip = int16(50);
saveNum = idivide(maxIter, saveSkip);
saveVar = zeros(saveNum, p);

currU = projectBoxSimplex(currX*H, D);

%% Main Loop: Frank-Wolfe Algorithm
for k = 1:maxIter
    
    % Save every (saveSkip) variable value (u)
    if mod(k, saveSkip) == 1
        saveVar(idivide(k, int16(saveSkip))+1, :) = currX;
    end
    
    % Obtain optimal u for current X
    u = projectBoxSimplex(currX*H, D);
    
    % Find all maximum indices of Hu
    evalU = H*u';
    evalMax = max(evalU);
    posMax = find(evalMax - evalU < tol);
    numMax = length(posMax);
    
    % Calculate s
    s = zeros(1, p);
    s(posMax) = 1/numMax;
    
    % Stopping criterion
    if (s-currX)*evalU < epsilon 
        break
    end
    
    % Update next iterate.
    % Strategies: use gamma = 2/(k+2), or use line search.
    if searchType == 0
        gamma = 2/(k+2);
        currX = (1-gamma).*currX + gamma.*s;
    elseif searchType == 1
        lsX = @(x, y, g) (1-g)*x + g*y;
        lsFn = @(g) - projectBoxSimplex(lsX(currX, s, g)*H, D)*H'*lsX(currX, s, g)';
        gamma = fminbnd(lsFn, 0, 1);
        currX = (1-gamma).*currX + gamma.*s;
    else
        lsX = @(x, y, g) (1-g)*x + g*y;
        lsFn = @(g) - projectBoxSimplex(lsX(currX, s, g)*H, D)*H'*lsX(currX, s, g)';
        gamma = fminbnd(lsFn, 0, 1);
        currX = (1-gamma).*currX + gamma.*s;
        
        evalS = evalU;
        evalU = H*currU';
        lsFn = @(g) max((1-g).*evalU + g.*evalS);
        gamma = fminbnd(lsFn, 0, 1);
        currU = (1-gamma).*currU + gamma.*u; %.*projectBoxSimplex(currX*H, D);

    end
    
end

%% Finish up

if k == maxIter
    exitflag = 0;
    if dbg
        fprintf('\n(Frank-Wolfe) Max iterations reached before convergence!\n');
    end
else
    exitflag = 1;
end

u = currU;
beta = max(H*currU');

end



function u = projectBoxSimplex(grad, D)
    % projections of linearised minimisers onto box constrained simplex
    % This is achieved by filling the best numfill directions with D.
    N = length(grad);
    numfill = floor(1/D);
    remainder = 1 - numfill*D;
    [~,ordered] = sort(grad);

    % If !int(1/D) we also have a remainder, r. If not the below is ok since r=0
    u = zeros(1, N);
    u(ordered(1:(numfill+1))) = [repmat(D, 1, numfill) remainder];
end
