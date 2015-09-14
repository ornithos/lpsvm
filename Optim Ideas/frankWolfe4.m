function [u, beta, exitflag] = frankWolfe4(H, initial, D, maxIter, epsilon, ...
    tol, searchType, dbg)
% Frank-Wolfe solution of the minimax problem for LPSVM. Using proximal
% ideas from R. Freund and Y. Nesterov.
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
if ~all(size(initial) == [1, N])
    error('(Frank-Wolfe) initial vector is of size [%d,%d]. Should be [1,%d].\n', ...
        size(initial,1), size(initial,2), N);
end
currU = initial;
lb = -Inf;

% Setup: Save intermediate steps for debug purposes.
saveSkip = int16(50);
saveNum = idivide(maxIter, saveSkip);
saveVar = zeros(saveNum, N);

% optimisation admin
mu = 1;                 % See Nesterov p60 (Def 2.1.2)  => CHECK THIS!


%% Main Loop: Frank-Wolfe Algorithm
for k = 1:maxIter
    
    % Save every (saveSkip) variable value (u)
    if mod(k, saveSkip) == 1
        saveVar(idivide(k, int16(saveSkip))+1, :) = currU;
    end
    
    % Compute the (prox) gradient of the max function wrt u.
    g = currU*H';
    x = exp(g./mu);
    x = x./sum(x);
    grad = x*H;
    
    % Compute new proposal direction (point) inside feasible region
    s = projectBoxSimplex(grad, D);
    
    % Calculate corresponding lower bound
    lb = max(lb, g*x' + grad*(s-currU)');
    lb = max(lb, max(g' + H*(currU-s)'));
    % Stopping criterion
    if grad * (currU - s)' < epsilon
        break
    end
    
    % Update next iterate.
    % Strategies: use gamma = 2/(k+2), or use line search.
    if searchType == 0
        gamma = 2/(k+2);
    else
        evalU = H*currU';
        evalS = H*s';
        lsFn = @(g) max((1-g).*evalU + g.*evalS);
        gamma = fminbnd(lsFn, 0, 1);
    end
    currU = (1-gamma).*currU + gamma.*s;
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
