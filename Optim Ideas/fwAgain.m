function [u, beta, exitflag] = fwAgain(H, initial, D, maxIter, epsilon, ...
    tol, searchType, dbg)
% Frank-Wolfe solution of the minimax problem for LPSVM
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

% Setup: Save intermediate steps for debug purposes.
saveSkip = int16(50);
saveNum = idivide(maxIter, saveSkip);
saveVar = zeros(saveNum, N);


%% Main Loop: Frank-Wolfe Algorithm
for k = 1:maxIter
    
    % Save every (saveSkip) variable value (u)
    if mod(k, saveSkip) == 1
        saveVar(idivide(k, int16(saveSkip))+1, :) = currU;
    end
    
    % Find maximum index of Hu
    evalU = H*currU';
    evalMax = max(evalU);
       
    % Subgradients: Test if more than one max of Hu
    posMax = find(evalMax - evalU < tol);
    numMax = length(posMax);
    if numMax == 1
        % Differentiable case - use projected gradient directly
        localGrad = H(posMax,:);
        s = projectBoxSimplex(localGrad,D);
    elseif numMax > 1
        % Subgradient case - choose subgradient that maximises inner prod
        if dbg
            fprintf('\n(Frank-Wolfe) Subdifferential case active..\n');
        end

        % search for steepest minimax subgradient in subdifferential hull
        tmp = quadprog(-0.5*[zeros(N,N), H(posMax,:)'; H(posMax,:), zeros(numMax,numMax)], ...
            [zeros(N,1); H(posMax,:)*currU'], [], [], ...
            [ones(1,N) zeros(1,numMax); zeros(1,N) ones(1,numMax)], ...
            ones(2,1), zeros(N+numMax,1), [D*ones(N,1); ones(numMax,1)], [], ...
            optimoptions('quadprog', 'Display','None'));

        localGrad = tmp((N+1):(N+numMax))'*H(posMax,:);
        s = projectBoxSimplex(tmp(1:N), D);   
    else
        error('\n(Frank-Wolfe) No hypotheses achieving maximum (?)\n');
    end
    
    
    % Stopping criterion
    if abs(localGrad * (currU - s)') < epsilon
        break
    end
    
    % Update next iterate.
    % Strategies: use gamma = 2/(k+2), or use line search.
    if searchType == 0
        gamma = 2/(k+2);
    else
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
