function [primal, dual, fval, exitflag] = ADMMDirect4(H, D, rho, eta, maxIter, ...
                        tol, reltol, x0, dualreq, verbose)
% ADMMDirect: Solve LP via ADMM using matrix factorisations.
%   See Boyd & Parikh chapter 5. The idea is to solve f^T x s.t. A*x = b
%   and sum(x) = 1 according to an unconstrained linear system. 
%
%   v4.
%   -- Accelerated as per Goldstein et al. (Accelerated Weakly Convex ADMM)
%   -- Primal version
%   -- factorisation of M'M cached (10-20x speedup)
%   -- using linsolve rather than mldivide (further 2-3x speedup)
%   -- **RHO CANNOT BE UPDATED SINCE REQUIRE NEW CHOLESKY
%   FACTORIZATION AND NEW RHS. NOT DIFFICULT BUT NOT IMPLEMENTED **
%
%   Arguments:
%   H          -   (Matrix) inequality constraint corr. to Hu <= beta
%   D          -   (Scalar) upper box constraint
%   rho        -   (Scalar) quadratic penalty constant in augm. lagrangian
%   eta        -   (Scalar) restart parameter. Restarts accel. if less than
%                  [1-eta] improvement in c. Suggest setting to 0.999.
%   maxIter    -   (Scalar) maximum number of iterations in ADMM loop
%   tol        -   (Scalar) stopping criterion.
%   reltol     -   (Scalar) relative convergence component.
%   x0         -   (Vector) warm start option. Can be smaller but not
%                  greater in length than sizeX (padded with 0s). Currently
%                  a dud.
%   dualreq    -   (Logical) Recalculate dual variables? Use for precision.
%
%   Outputs:
%   primal     -   Primal variables. Feasible but not necessarily optimal.
%                  Margin is approximately correct, but H'a + xi may be less
%                  than margin for a few datapoints.
%   dual       -   Dual Variables. Again, are only approximate, and some Hu
%                  may exceed beta. Null is dualreq is false.
%   exitflag   -   1  = converged, 0 = number of iterations exceeded,
%                  -1 = iterations exceeded / infeasibility > 10*tol.

%% set-up
% ensuring correct orientation of vectors
if any(size(D))>1; error('D must be scalar'); end;
if eta >= 1 || eta < 0.25; error('eta must be less than, but close to 1'); end;
    
% create arrays for f(x): Least Squares
[p, n]         = size(H);
f              = [D*ones(n,1); zeros(n,1); -1; +1; zeros(p,1)];
b              = [zeros(n,1); 1];
A              = [eye(n), -eye(n), -ones(n,1), ones(n,1), H'; zeros(1,2*n+2), ones(1,p)];
sizeX          = 2*n+p+2;
constraintM    = [rho*eye(sizeX), A'; A, zeros(n+1)];

% Initialise ADMM variables
if ~isempty(x0)
    extralength = sizeX - length(x0);
    if size(x0,2) > 1; x0 = x0'; end;
    
    if extralength < 0
        error('x0 must not be greater in size than the vector x')
    elseif extralength > 0
        x0 = [x0; zeros(extralength,1)];
    end
    zOld = x0;
    zHat = x0;
else
    zOld           = zeros(sizeX,1);
    zHat           = zeros(sizeX,1);
end
uOld           = zeros(sizeX,1);
uHat           = zeros(sizeX,1);
c              = [0, Inf];              % initialise 'restart' tracker c.
alpha          = [1, 1];
restarts       = 0;

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
opts.LT = true; opts.TRANSA = false;     % (linsolve) triangular backsolve
optsT.LT = true; optsT.TRANSA = true;    % (linsolve) Transposed tri backsolve

for k = 1:maxIter

    % x-update
    % RHS = constraintM * [ rho*(z - u) - f; b ];
    RHS = baseRHS + addlRHS(rho, zHat, uHat, H, n, p);
    % -- Solve linear system using Cholesky -----
        w   = linsolve(cholM,RHS,opts);
        tmp = linsolve(cholM,w,optsT);
    % -------------------------------------------
    x   = tmp(1:sizeX);
    
    % more basic updates
    z     = max(x + uHat, 0);                  % non-neg projection for z
    u     = uHat + x - z;                      % dual update
    
    uMu   = u - uHat;
    zMz   = zHat - z;                          % opp. way around due to B. (redundant)
    c(1)  = uMu'*uMu + zMz'*zMz;
    saveX(1:sizeX,k) = z;
    
    % produce Nesterov-like accelerated estimates.
    if c(1) < eta*c(2)
        alpha(1) = (1 + sqrt(1 + 4*alpha(2)^2))/2;
        a        = (alpha(2)-1)/alpha(1);
        zHat     = (1 + a)*z -a*zOld;
        uHat     = (1 + a)*u -a*uOld;
    else
        alpha(1) = 1;
        zHat     = z;        % Changed/I think this is a misprint in Goldstein.
        uHat     = u;
        c(1)     = c(2)/eta;
        restarts  = restarts + 1;
    end
    c(2) = c(1);
    alpha(2) = alpha(1);
    
    % diagnostics, reporting, termination checks
    history.objval(k)  = f'*x;
    history.r_norm(k)  = norm(x - z);
    history.s_norm(k)  = norm(-rho*(z - zOld));

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

    zOld = z;
    uOld = u;
end

%% Output
saveX(sizeX+1,1:k) = history.objval;
fprintf('num iter: %d, fpval: %1.7f, rho: %1.7f, restarts: %d\n', k, ...
    history.objval(k), [zeros(2*n,1); -1; +1; zeros(p,1)]'*z, restarts);

fval       = history.objval(k);
exitflag   = 1;

if k == maxIter
    exitflag = 0;
    if (history.r_norm(k) > 10*history.eps_pri(k) || ...
            history.s_norm(k) > 10*history.eps_dual(k))
        exitflag = -1;
    end
end

% Assign primal/dual variables
primal.xi  = z(1:n);
primal.a   = z((end-p+1):end);
primal.rho = z(2*n+1) - z(2*n+2);

%% Recalulate Dual Variables using KKT System. While dual variables
%  technically converge, this happens too slowly to be useful. Here we
%  assume that the (i) active set (u_i > 0) has been found, (ii) the
%  bounded vectors are known (u_i == D), and (iii) the optimal beta is
%  known (fval), since the objective val converges quickly. The resulting
%  system is used to calculate the (0 < u_i < D) variables exactly.

if dualreq
    viol    = z(1:n) > 1e-5;
    nm      = sum(viol);

    J       = primal.a > 1e-5;
    nJ      = sum(J);
    reqd    = ceil((1-D*nm)/D);
    avail   = nJ;
    if reqd >  avail
        error(['KKT dual solve failure - more indices to be determined than ', ...
            'constraints (%d > %d)'], reqd, avail);
    end

    u       = zeros(n,1);
    u(viol) = D;

    if reqd >= 1
        gap     = abs(z((n+1):2*n) + z(1:n));
        [~,go]  = sort(gap);
        free    = go(1:avail);
        M       = [H(J,free); ones(1, avail)];
        b       = [-fval*ones(avail,1) - D*sum(H(J,viol),2); 1 - nm*D];
        freeval = lsqnonneg(M,b);

        u(free) = freeval;
    end


    dual.u     = u;
    dual.beta  = -fval;
else
    dual       = [];
end
end


% Additional (Variable) part of RHS term updated with new (z, u)
% Note that changing rho requires much more than this
function out = addlRHS(rho, z, u, H, n, p)
zmu = z - u;
a = zmu((end-p+1):end);
out = [rho^2.*zmu;
       rho.*(zmu(1:n) - zmu((n+1):(2*n)) - (zmu(2*n+1)-zmu(2*n+2))*ones(n,1) + H'*a);
       rho.*sum(a)];
end