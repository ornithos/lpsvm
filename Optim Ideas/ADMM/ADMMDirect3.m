function [z, history] = ADMMDirect3(H, D, rho, alpha, maxIter, ...
                        tol, reltol, verbose)
% ADMMDirect: Solve LP via ADMM using matrix factorisations.
%   See Boyd & Parikh chapter 5. The idea is to solve f^T x s.t. A*x = b
%   and sum(x) = 1 according to an unconstrained linear system. 
%
%   v3. Here we attack the primal rather than the dual (for parallelisation
%   purposes).
%   -- factorisation of M'M cached (10-20x speedup)
%   -- using linsolve rather than mldivide (further 2-3x speedup)
%
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
if any(size(D))>1; error('D must be scalar'); end;
if alpha > 1.8 || alpha < 1.0; fprintf('alpha not in suggested range\n'); end;

% Constants
RESCALE_RHO    = true;
MU_TRIGGER     = 10;     % Diff in norms required to trigger rescaling rho
TAU_INCR       = 10;
TAU_DECR       = 2;

% create arrays
[p, n]         = size(H);
f              = [D*ones(n,1); zeros(n,1); -1; +1; zeros(p,1)];
b              = [zeros(n,1); 1];
A              = [eye(n), -eye(n), -ones(n,1), ones(n,1), H'; zeros(1,2*n+2), ones(1,p)];
sizeX          = 2*n+p+2;
constraintM    = [rho*eye(sizeX), A'; A, zeros(n+1)];
z              = zeros(sizeX,1);     % Initialise (z, u) to 0.
u              = zeros(sizeX,1);

saveX = zeros(sizeX+1,maxIter);
%% Main ADMM Section
% following code adapted from S. Boyd, N. Parikh, E. Chu, B. Peleato, 
% and J. Eckstein: https://web.stanford.edu/~boyd/papers/admm/linprog/linprog.html

if verbose
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

% Setup for fast linear solve with cached factorisation
cholM = chol(constraintM*constraintM)';
baseRHS = [(-rho*f + [zeros(2*n+2,1); ones(p,1)]); -(D+2)*ones(n,1); 0];
tmp = constraintM \ [ rho*(z - u) - f; b ];

for k = 1:maxIter

    % x-update
        RHS = baseRHS + addlRHS(rho, z, u, H, n, p);
%     if max(abs(RHS-RHS2)) > 1e-14
%         fprintf('RHS Failure');
%     end
    opts.LT = true; opts.TRANSA = false;
    w   = linsolve(cholM,RHS,opts);
    opts.LT = true; opts.TRANSA = true;
    tmp = linsolve(cholM,w,opts);
%     w   = cholM\RHS;
%     tmp = cholM'\w;
    x   = tmp(1:sizeX);
    
%     tmp = constraintM \ [ rho*(z - u) - f; b ];
%     x2   = tmp(1:sizeX);
% 
%     if max(abs(x-x2)>1e-6)
%         stophere=1;
%     end
    
    % z-update with relaxation
    zold  = z;
    x_hat = alpha*x + (1 - alpha)*zold;        % relaxation
    z     = max(x_hat + u, 0);                 % non-neg projection for x
    saveX(1:sizeX,k) = z;
    
    % dual update
    u = u + (x_hat - z);

    % diagnostics, reporting, termination checks
    history.objval(k)  = f'*x;
    history.r_norm(k)  = norm(x - z);
    history.s_norm(k)  = norm(-rho*(z - zold));

    history.eps_pri(k) = sqrt(sizeX)*tol + reltol*max(norm(x), norm(-z));
    history.eps_dual(k)= sqrt(sizeX)*tol + reltol*norm(rho*u);

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
            fprintf('iter %d: multiply rho by %d\n', k, TAU_INCR);
            rho = TAU_INCR * rho;
            u = u / TAU_INCR;
            constraintM    = [rho*eye(sizeX), A'; A, zeros(n+1)];
        elseif history.s_norm(k) > MU_TRIGGER * history.r_norm
            fprintf('iter %d: divide rho by %d\n', k, TAU_DECR);
            rho = rho / TAU_DECR;
            u = u * TAU_DECR;
            constraintM    = [rho*eye(sizeX), A'; A, zeros(n+1)];
        end
    end
end
saveX(sizeX+1,1:k) = history.objval;
fprintf('num iter: %d, fpval: %1.7f, rho: %1.7f\n', k, history.objval(k), ...
    [zeros(2*n,1); -1; +1; zeros(p,1)]'*z);
end

function out = addlRHS(rho, z, u, H, n, p)
zmu = z - u;
a = zmu((end-p+1):end);
out = [rho^2.*zmu;
       rho.*(zmu(1:n) - zmu((n+1):(2*n)) - (zmu(2*n+1)-zmu(2*n+2))*ones(n,1) + H'*a);
       rho.*sum(a)];
end