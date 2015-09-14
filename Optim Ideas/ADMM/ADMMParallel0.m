function [z, history] = ADMMParallel1(H_all, P, D, rho, alpha, maxIter, ...
                        tol, reltol, verbose)
% ADMMDirect: Solve LP via ADMM using matrix factorisations.
%   See Boyd & Parikh chapter 5. The idea is to solve f^T x s.t. A*x = b
%   and sum(x) = 1 according to an unconstrained linear system. 
%
%   v3. Here we attack the primal rather than the dual (for parallelisation
%   purposes).

%   Arguments:
%   H_all      -   (Matrix) Entire data matrix H
%   P          -   (Cell) Partition of data - each element i contains the
%                   indices of the data on 'node' i.
%   D          -   (Scalar) upper box constraint
%   rho        -   (Scalar) quadratic penalty constant in augm. lagrangian.
%                   Note this corresponds to the gradient step in dual.
%   alpha      -   (Scalar) relaxation parameter, choose [1.5, 1.8] for speedup
%   maxIter    -   (Scalar) maximum number of iterations in ADMM loop
%   tol        -   (Scalar) stopping criterion.
%   reltol     -   (Scalar) additional slack from relative convergence criteria

%% set-up
% ensuring correct orientation of vectors
if any(size(D))>1; error('D must be scalar'); end;
if alpha > 1.8 || alpha < 1.0; fprintf('alpha not in suggested range\n'); end;

% Constants
RESCALE_RHO    = true;
MU_TRIGGER     = 10;     % Diff in norms required to trigger rescaling rho
TAU_INCR       = 10;
TAU_DECR       = 2;

[p, n]         = size(H);
nu             = 1/(n*D);

% Setup random partition if user not specified
nP = numel(P);
if ~iscell(P)
    if nP == 1;
        nP = P;
        P = cell(nP,1);
        rp = randperm(n);
        pns = [floor(n/nP), mod(n,nP)];
        pns = pns(1)*ones(nP,1) + [ones(pns(2),1); zeros(nP-pns(2),1)];
        pns = [0; pns(randperm(nP))];
        for i = 1:nP
            P{i} = rp((pns(i)+1):pns(i+1));
        end
    else
        error('P must be either a data partition or a scalar for number of partitions.');
    end
end

    
% create arrays
M  = cell(nP,1);
f  = cell(nP,1);
b  = cell(nP,1);
lb = cell(nP,1);
ub = cell(nP,1);
for i = 1:nP
    cN    = size(H_all{i}, 2);
    M{i}  = [H_all(:,P{i})', eye(cN), -ones(cN,1), ones(cN,1); ones(1,p), zeros(cN+1,1)];
    f{i}  = [zeros(p,1); (1/nu).*ones(cN,1); -cN, +cN];
    b{i}  = [zeros(cN,1); 1];
    lb{i} = [zeros(cN+p,1); -Inf];
    ub{i} = [(1/nu).*ones(cN+p,1); Inf];
end


sizeX          = 2*n+p+2;
constraintM    = [rho*eye(sizeX), A'; A, zeros(n+1)];
z              = zeros(sizeX,1);     % Initialise (z, u) to 0.
u              = zeros(sizeX,1);

saveX = zeros(sizeX,maxIter);
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
    x   = tmp(1:sizeX);

    % z-update with relaxation
    zold  = z;
    x_hat = alpha*x + (1 - alpha)*zold;        % relaxation
    z     = max(x_hat + u, 0);                 % non-neg projection for x
    saveX(:,k) = z;
    
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
fprintf('num iter: %d\n', k);
end