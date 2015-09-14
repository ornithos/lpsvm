function [primal, dual, fval, exitflag] = ADMMDistributed6(H, D, rho, eta, ...
                        maxIter, tol, reltol, x0, npar, dualreq, verbose)
% ADMMDistributed: Solve LP via ADMM using distributed version of
%   Boyd & Parikh chapter 5. The idea is to solve f^T x s.t. A*x = b
%   and sum(x) = 1 according to an unconstrained linear system. The linear
%   system can be solved in parallel for each partition of data, and the
%   non-negativity and averaging is computed in a centralised step.
%
%   v6
%   -- Nesterov Accelerated ** THIS REALLY IS NOT AN IMPROVEMENT
%   -- ** convergence is so noisy that 'momentum' cannot help **
%   -- solving x via ADMM (KKT) iterations.
%   -- Primal version of ADMM using consensus across nodes.
%   -- Excluding xi from consensus entirely. 
%   -- using quadprog to solve QP.
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

[p, n]        = size(H);         % n is total variables across partitions
par           = cell(npar,1);    % Parallel parameters stored in par
D2            = n*D;             % rescale D, and later rho.

%% Partition data
[P, np, Hpar] = partitionData(H, npar);

for i = 1:npar
% create arrays for f(x): Least Squares
% note that f is scaled by n
    par{i}.n     = np(i);
    par{i}.H     = Hpar{i};
    par{i}.idx   = P{i};
    par{i}.f     = [D2*ones(np(i),1); zeros(np(i),1); -par{i}.n; zeros(p,1)]; 
    par{i}.A     = [eye(np(i)), -eye(np(i)), -ones(np(i),1), par{i}.H';
                    zeros(1,2*np(i)+1), ones(1,p)];
    par{i}.b     = [zeros(np(i),1); 1];
    par{i}.sX    = np(i)*2+p+1;
    par{i}.savX  = NaN(par{i}.sX+2+(2*p+2),maxIter);
    
    par{i}.F     = eye(par{i}.sX);
    par{i}.F(1:np(i),1:np(i)) = zeros(np(i));
    par{i}.M     = [rho*eye(par{i}.sX), par{i}.A'; par{i}.A, zeros(np(i)+1)];
    
    par{i}.z     = [zeros(1,1); zeros(par{i}.n*2, 1); ones(p,1)./p];
    par{i}.zHat  = [zeros(1,1); zeros(par{i}.n*2, 1); ones(p,1)./p];
    par{i}.zOld  = [zeros(1,1); zeros(par{i}.n*2, 1); ones(p,1)./p];
    par{i}.u     = zeros(par{i}.sX, 1);
    par{i}.uHat  = zeros(par{i}.sX, 1);
    par{i}.uOld  = zeros(par{i}.sX, 1);
    
    par{i}.gl_ix = (np(i)*2+1):(np(i)*2+1+p);
end

saveX          = zeros(p+1+n+1+npar+1,maxIter);

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

% Nesterov Variables
c              = [0, Inf];              % initialise 'restart' tracker c.
alpha          = [1, 1];
restarts       = 0;

for k = 1:maxIter

    % reset diagnostic summations (quantities used as running sums)
    d.Fx     = 0;
    d.X      = 0;
    d.Z      = 0;
    d.U      = 0;
    d.XmZSq  = 0;
    d.ZmZSq  = 0;
    
    
    % 'PARALLEL' x-update
    for i = 1:npar
        % Update x 
        tmp = par{i}.M \ [ rho*(par{i}.zHat - par{i}.uHat) - par{i}.f; par{i}.b];
        par{i}.x   = tmp(1:par{i}.sX);
    end

    % GATHER: z-update (centralised subset, (a, rho))
    gl_Z        = globalMeanZ(par, p);
    
    % setup calcs for Nesterov
    uDelta2 = 0;
    zDelta2 = 0;
    
    % 'PARALLEL': z-update (local subset (xi))
    for i = 1:npar
        par{i}.zOld           = par{i}.z;
        
        par{i}.z(1:(end-p-1)) = max(par{i}.x(1:(end-p-1)) + ...
                                    par{i}.uHat(1:(end-p-1)), 0);
        par{i}.z((end-p):end) = gl_Z;
        par{i}.u              = par{i}.u - par{i}.z + par{i}.x;
        
        % calcs for Nesterov
        uMu                   = par{i}.u - par{i}.uHat;
        zMz                   = par{i}.z - par{i}.zHat;
        uDelta2               = uDelta2 + sum(uMu.*uMu);
        zDelta2               = zDelta2 + sum(zMz.*zMz);        
    end
    
    % Nesterov Branch (essentially check if possible)
    c(1) = uDelta2 + zDelta2;
    if c(1) < eta*c(2)
        alpha(1) = (1 + sqrt(1 + 4*alpha(2)^2))/2;
        a        = (alpha(2)-1)/alpha(1);
    else
        alpha(1) = 1;
        a        = 0;        % Changed/I think this is a misprint in Goldstein.
        c(1)     = c(2)/eta;
        restarts = restarts + 1;
    end
    
    % Nesterov Estimates
    for i = 1:npar
        par{i}.zHat = (1 + a)*par{i}.z - a*par{i}.zOld;
        par{i}.uHat = (1 + a)*par{i}.u - a*par{i}.uOld;
    end
        
    % >>>> for diagnostics
    for i = 1:npar
        
        saveX((p+1)+P{i},k) = par{i}.x(1:par{i}.n);
        d.Fx        = d.Fx + par{i}.f'*par{i}.x;
        
        saveX(p+n+2+i,k) = par{i}.f'*par{i}.x;
        par{i}.savX(1:par{i}.sX,k) = par{i}.x;
        par{i}.savX((par{i}.sX+2):(par{i}.sX+2+p),k) = gl_Z;
        par{i}.savX((par{i}.sX+2+p+2):(par{i}.sX+2+2*p+2),k) = par{i}.u((end-p):end);
        
        d.X         = d.X + sum(par{i}.x((par{i}.n+1):end).^2);
        d.Z         = d.Z + sum(gl_Z.^2);
        d.U         = d.U + sum(par{i}.u.^2);
        xmz         = (par{i}.x - (par{i}.z));
        d.XmZSq     = d.XmZSq + sum(xmz.*xmz);
        zmz         = par{i}.z - par{i}.zOld;
        d.ZmZSq     = d.ZmZSq + sum(zmz.*zmz);
    end 
        
    saveX(1:(p+1),k) = gl_Z;
    if k == 120
        stophere = 0;
    end
    
    % save diagnostics history, check termination conditions.
    history.objval(k)  = d.Fx/n;       % all obj scaled by n in nodes
    history.r_norm(k)  = sqrt(d.XmZSq);
    history.s_norm(k)  = rho*sqrt(d.ZmZSq);

    history.eps_pri(k) = sqrt(p+1)*tol + reltol*max(sqrt(d.X), sqrt(d.Z));
    history.eps_dual(k)= sqrt(p+1)*tol + reltol*rho*sqrt(d.U);

    if verbose
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end
    
    % Update Nesterov Variables for next iteration
    c(2) = c(1);
    alpha(2) = alpha(1);
    for i=1:npar
        par{i}.zOld = par{i}.z;
        par{i}.uOld = par{i}.u;
    end
end

%% Output
saveX(p+n+2,1:k) = history.objval;
fprintf('num iter: %d, fpval: %1.7f, rho: %1.7f, restarts: %d\n', k, ...
    history.objval(k), gl_Z(1), restarts);

fval       = history.objval(k);
exitflag   = 1;

if k == maxIter
    exitflag = 0;
    if (history.r_norm(k) > 10*history.eps_pri(k) || ...
            history.s_norm(k) > 10*history.eps_dual(k))
        exitflag = -1;
    end
end

% construct z

% Assign primal/dual variables
primal.a   = gl_Z(2:end);
primal.rho = gl_Z(1);
primal.xi  = max(primal.rho - H'*primal.a,0);
%primal.a   = zglobal(3:end);
%primal.rho = zglobal(1) - zglobal(2);

%% Recalulate Dual Variables using KKT System. While dual variables
%  technically converge, this happens too slowly to be useful. Here we
%  assume that the (i) active set (u_i > 0) has been found, (ii) the
%  bounded vectors are known (u_i == D), and (iii) the optimal beta is
%  known (fval), since the objective val converges quickly. The resulting
%  system is used to calculate the (0 < u_i < D) variables exactly.

if dualreq
    viol    = primal.xi > 1e-5;
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
        [~,xio]  = sort(primal.xi);
        free    = xio(1:avail);
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



function gl_Z = globalMeanZ(pars, p)
% GATHER: z-update (centralised subset, (a, rho))
cumSum = zeros(p+1,1);
N = length(pars);
for i = 1:N
    cumSum = cumSum + pars{i}.x(pars{i}.gl_ix);
    cumSum = cumSum + pars{i}.uHat(pars{i}.gl_ix);
end
gl_Z        = cumSum./N;
gl_Z(2:end) = max(gl_Z(2:end),0);   % rho does not have to be positive.
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
% % BELOW IS FOR ACCELERATED VARIANT
% function out = addlRHS(rho, z, u, zloc, uloc, H, n, p)
% z   = [zloc; z];
% u   = [uloc; u];
% zmu = z - u;
% a = zmu((end-p+1):end);
% out = [rho^2.*zmu;
%        rho.*(zmu(1:n) - zmu((n+1):(2*n)) - (zmu(2*n+1)-z(2*n+2))*ones(n,1) + H'*a);
%        rho.*sum(a)];
% end


% Partition the data into npar blocks.
function [P, pns, Hout] = partitionData(H, npar)
rand('state', 0);
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