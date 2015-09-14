function [primal, dual, fval, exitflag] = ADMMDistributed1(H, D, rho, eta, ...
                        maxIter, tol, reltol, x0, npar, dualreq, verbose)
% ADMMDistributed: Solve LP via ADMM using distributed version of
%   Boyd & Parikh chapter 5. The idea is to solve f^T x s.t. A*x = b
%   and sum(x) = 1 according to an unconstrained linear system. The linear
%   system can be solved in parallel for each partition of data, and the
%   non-negativity and averaging is computed in a centralised step.
%
%   v1
%   -- Accelerated primal version of ADMM for LPs using Cholesky cache and
%      linsolve.
%   -- 'Parallel' element computed in serial.
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
%   x0         -   (Vector) warm start option. (DEPRECATED)
%   npar       -   (Scalar) Number of partitions / nodes to simulate.
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

[p, ~]        = size(H);         % n is variables over partitions
par           = cell(npar,1);    % Parallel parameters stored in par


%% Partition data
[P, np, Hpar] = partitionData(H, npar);

for i = 1:npar
% create arrays for f(x): Least Squares
    par{i}.n    = np(i);
    par{i}.H    = Hpar{i};
    par{i}.idx  = P{i};
    par{i}.f    = [D*ones(np(i),1); zeros(np(i),1); -1; +1; zeros(p,1)];
    par{i}.b    = [zeros(np(i),1); 1];
    par{i}.A    = [eye(np(i)), -eye(np(i)), -ones(np(i),1), ones(np(i),1), ...
                   par{i}.H'; zeros(1,2*np(i)+2), ones(1,p)];
    par{i}.sX   = 2*np(i)+p+2;
    par{i}.M    = [rho*eye(par{i}.sX), par{i}.A'; par{i}.A, zeros(np(i)+1)];
    
    par{i}.chol = chol(par{i}.M*par{i}.M)';
    par{i}.bRHS = [(-rho*par{i}.f + [zeros(2*np(i)+2,1); ones(p,1)]);
                   -(D+2)*ones(np(i),1); 0];
    par{i}.z    = zeros(np(i)*2, 1);
    par{i}.u    = zeros(np(i)*2, 1);
    par{i}.zOld = zeros(np(i)*2, 1);
    par{i}.zHat = zeros(np(i)*2, 1);
    par{i}.uOld = zeros(np(i)*2, 1);
    par{i}.uHat = zeros(np(i)*2, 1);
    par{i}.cloc    = [0, Inf];              % initialise 'restart' tracker c.
%     par{i}.alpha= [1, 1];
%     par{i}.rstt = 0;
end

% Global ADMM variables, corresponding to 'a' and margin (primal 'rho').
zOld           = zeros(p+2,1);
zHat           = zeros(p+2,1);
uOld           = zeros(p+2,1);
uHat           = zeros(p+2,1);
c              = [0, Inf];              % initialise 'restart' tracker c.
a              = 0;
alpha          = [1, 1];
restarts       = 0;

saveX = zeros(p+3,maxIter);
%% Main ADMM Section
% following code adapted from S. Boyd, N. Parikh, E. Chu, B. Peleato, 
% and J. Eckstein: https://web.stanford.edu/~boyd/papers/admm/linprog/linprog.html

if verbose
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

% Setup options for linsolve (Left Triangular System)
opts.LT = true; opts.TRANSA = false;     % (linsolve) triangular backsolve
optsT.LT = true; optsT.TRANSA = true;    % (linsolve) Transposed tri backsolve

for k = 1:maxIter

    % 'PARALLEL' x-update
    for n = 1:npar
        
        % Nesterov acceleration (kind of previous step)
        par{i}.zHat = (1 + a)*par{i}.z - a*par{i}.zOld;
        par{i}.uHat = (1 + a)*par{i}.u - a*par{i}.uOld;
        
        % Update x (this step
        par{i}.RHS = par{i}.bRHS + addlRHS(rho, zHat, uHat, par{i}.zHat, ...
            par{i}.uHat, par{i}.H, par{i}.n, p);
        % -- Solve linear system using Cholesky -----
            w   = linsolve(par{i}.chol, par{i}.RHS, opts);
            tmp = linsolve(par{i}.chol, w, optsT);
        % -------------------------------------------
        par{i}.x   = tmp(1:par{i}.sX);
    end
   
    
    % 'PARALLEL' local subset of z/u - update (xi, w).
    for n = 1:npar
        par{i}.xloc = par{i}.x(1:(par{i}.n*2));
        par{i}.z    = max(par{i}.xloc + par{i}.uHat, 0);           
        par{i}.u    = par{i}.uHat + par{i}.xloc - par{i}.z;
        uMu         = par{i}.u - par{i}.uHat;
        zMz         = par{i}.zHat - par{i}.z;
        par{i}.cloc = uMu'*uMu + zMz'*zMz;
    end
    
    
    % GATHER: Centralised subset of z/u - update (a, rho)
    ztmp  = zeros(p+2,1);
    ctmp = 0;
    for n = 1:npar
        ztmp = ztmp + par{i}.x((par{i}.n*2 + 1):end);  % GATHER rho, a.
        ctmp = ctmp + par{i}.cloc;                     % pre-calcd local c
    end
    
    z     = max(ztmp + uHat, 0);               % non-neg projection for z
    u     = uHat + x - z;                      % dual update
    
    
    
    % calculate / gather c for Nesterov calcs
    uMu              = u - uHat;
    zMz              = zHat - z;
    c(1)             = ctmp + uMu'*uMu + zMz'*zMz;
    saveX(1:(p+2),k) = z;
    
    % Nesterov calcs for global and admin.
    if c(1) < eta*c(2)
        alpha(1) = (1 + sqrt(1 + 4*alpha(2)^2))/2;
        a        = (alpha(2)-1)/alpha(1);
        zHat     = (1 + a)*z -a*zOld;
        uHat     = (1 + a)*u -a*uOld;
    else
        alpha(1) = 1;
        a        = 0;
        zHat     = z;        % Changed/I think this is a misprint in Goldstein.
        uHat     = u;
        c(1)     = c(2)/eta;
        restarts  = restarts + 1;
    end
    c(2) = c(1);
    alpha(2) = alpha(1);
    zOld = z;
    uOld = u;
    
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
function out = addlRHS(rho, z, u, zloc, uloc, H, n, p)
z   = [zloc; z];
u   = [uloc; u];
zmu = z - u;
a = zmu((end-p+1):end);
out = [rho^2.*zmu;
       rho.*(zmu(1:n) - zmu((n+1):(2*n)) - (zmu(2*n+1)-z(2*n+2))*ones(n,1) + H'*a);
       rho.*sum(a)];
end


% Partition the data into npar blocks.
function [P, pns, Hout] = partitionData(H, npar)
n     = size(H,2);
rp    = randperm(n);                % randomly order elements
P     = cell(npar,1);
Hout  = cell(npar,1);
pns   = [floor(n/npar), mod(n,npar)]; % number of elements in each partition
pns   = pns(1)*ones(npar,1) + [ones(pns(2),1); zeros(npar-pns(2),1)];
pns   = pns(randperm(npar));         % add 0 at beginning for ease in loop.
cns   = [0; cumsum(pns)];            % randomisation is over num elemnts in
                                     % each partition when mod(n, npar) != 0
for i = 1:npar
    P{i}    = rp((cns(i)+1):cns(i+1));      % randomisation here is over el
    Hout{i} = H(:,P{i});
end
end