function [z, history] = ADMMDirect(A, b, rho, alpha, maxIter, ...
                        tol, reltol, verbose)
% ADMMDirect: Solve LP via ADMM using matrix factorisations.
%   See Boyd & Parikh chapter 5. The idea is to solve f^T x s.t. A*x = b
%   and sum(x) = 1 according to an unconstrained linear system. 
%
%   v2. Works from precalc'd quantities A and b rather than H. Also leaves
%   all constraints in A except positivity. So the projection step has
%   changed from box --> non-negativity. Turns out this didn't help.

%   Arguments:
%   H          -   (Matrix) inequality constraint corr. to Hu <= beta
%   D          -   (Scalar) upper box constraint
%   rho        -   (Scalar) quadratic penalty constant in augm. lagrangian
%   alpha      -   (Scalar) relaxation parameter, choose [1.5, 1.8] for speedup
%   maxIter    -   maximum number of iterations in ADMM loop
%   tol        -   stopping criterion.
%   reltol     -   additional slack from relative convergence criteria

%% set-up
% ensuring correct orientation of vectors
if size(b,2) > 1; b = b'; end;
if alpha > 1.8 || alpha < 1.5; fprintf('alpha not in suggested range\n'); end;

% Constants
RESCALE_RHO    = false;
MU_TRIGGER     = 10;     % Diff in norms required to trigger rescaling rho
TAU_INCR       = 2;
TAU_DECR       = 2;

% create arrays
[m, n]         = size(A);
f              = [zeros(n-2,1); +1; -1];
constraintM    = [rho*eye(n), A'; A, zeros(m)];
z              = zeros(n,1);     % Initialise (z, u) to 0.
u              = zeros(n,1);

%% Main ADMM Section
% following code adapted from S. Boyd, N. Parikh, E. Chu, B. Peleato, 
% and J. Eckstein: https://web.stanford.edu/~boyd/papers/admm/linprog/linprog.html

if verbose
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

for k = 1:maxIter

    % x-update
    tmp = constraintM \ [ rho*(z - u) - f; b ];
    x = tmp(1:n);

    % z-update with relaxation
    zold = z;
    x_hat = alpha*x + (1 - alpha)*zold;        % relaxation
    z = max(x_hat + u, 0);                     % non negativity constraint

    % dual update
    u = u + (x_hat - z);

    % diagnostics, reporting, termination checks
    history.objval(k)  = f'*x;
    history.r_norm(k)  = norm(x - z);
    history.s_norm(k)  = norm(-rho*(z - zold));

    history.eps_pri(k) = sqrt(n)*tol + reltol*max(norm(x), norm(-z));
    history.eps_dual(k)= sqrt(n)*tol + reltol*norm(rho*u);

    if verbose
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end
    
    % update rho?
    if RESCALE_RHO
        if history.r_norm(k) > MU_TRIGGER * history.s_norm
            fprintf('rho: %3.2f -> %3.2f\n', rho, TAU_INCR*rho);
            rho = TAU_INCR * rho;
            u = u / TAU_INCR;
            constraintM    = [rho*eye(sizeX), A'; A, zeros(m+1)];
        elseif history.s_norm(k) > MU_TRIGGER * history.r_norm
            fprintf('rho: %3.2f -> %3.2f\n', rho, rho / TAU_DECR);
            rho = rho / TAU_DECR;
            u = u * TAU_DECR;
            constraintM    = [rho*eye(sizeX), A'; A, zeros(m+1)];
        end
    end
end

fprintf('num iter: %d\n', k);
end
