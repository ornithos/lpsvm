function [primal, dual, fval, exitflag] = ADMMDistributed2(H, D, rho, eta, ...
                        maxIter, tol, reltol, x0, npar, dualreq, verbose)
% ADMMDistributed: Solve LP via ADMM using distributed version of
%   Boyd & Parikh chapter 5. The idea is to solve f^T x s.t. A*x = b
%   and sum(x) = 1 according to an unconstrained linear system. The linear
%   system can be solved in parallel for each partition of data, and the
%   non-negativity and averaging is computed in a centralised step.
%
%   v2
%   -- Primal version of ADMM using consensus across nodes.
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
    par{i}.n    = np(i);
    par{i}.H    = Hpar{i};
    par{i}.idx  = P{i};
    par{i}.f    = [D2*ones(np(i),1); -par{i}.n; zeros(p,1)];   % note scaling by n and attribution
    par{i}.A    = -[eye(np(i)), -ones(np(i),1), par{i}.H'];
    par{i}.b    = zeros(np(i),1);
    par{i}.Aeq  = [zeros(1,np(i)+1), ones(1,p)];
    par{i}.beq  = 1;
    par{i}.lb   = [zeros(1,np(i)), -Inf, zeros(1,p)];
    par{i}.sX   = np(i)+p+1;
    par{i}.z    = [zeros(1,1); zeros(par{i}.n, 1); ones(p,1)./p];
    par{i}.zOld = [zeros(1,1); zeros(par{i}.n, 1); ones(p,1)./p];
    par{i}.u    = zeros(par{i}.sX, 1);
    par{i}.savX = NaN(par{i}.sX*3+2 ,maxIter);
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
        [par{i}.x,~,ef] = quadprog((rho)*eye(par{i}.sX), par{i}.f + rho*...
            (par{i}.u - par{i}.z), par{i}.A, par{i}.b, par{i}.Aeq,  ...
            par{i}.beq, par{i}.lb, [], [], optimset('Display','None')); 
        if ef ~= 1
            stophere = 1;
        end
    end

    % GATHER: z-update (centralised subset, (a, rho))
    zglobal    = zeros(p+1,1);
    if k == 1
        for i = 1:npar
            zglobal = zglobal + par{i}.x((par{i}.n+1):end) ...
                    + par{i}.u((par{i}.n+1):end);       % GATHER z + u
        end
    else
        % average(u)=0 for each global var after 1st iter.
        for i = 1:npar
            zglobal = zglobal + par{i}.x((par{i}.n+1):end); % GATHER z
        end
    end
    zglobal    = zglobal./npar;   % avg (z+u)
    
    
    % 'PARALLEL' u-update and local subset of z (xi, w).
    for i = 1:npar
        par{i}.zOld = par{i}.z;
        
        % update z / u
        xlocal      = par{i}.x(1:(par{i}.n));
        par{i}.z    = [xlocal + par{i}.u(1:(par{i}.n)); zglobal];
        par{i}.u    = par{i}.u + par{i}.x - par{i}.z;
        
        % >>>> for diagnostics
        saveX((p+1)+P{i},k) = par{i}.z(1:par{i}.n);
        d.Fx        = d.Fx + par{i}.f'*par{i}.x;
        
        saveX(p+n+2+i,k) = par{i}.f'*par{i}.x;
        par{i}.savX(1:par{i}.sX,k) = par{i}.x;
        par{i}.savX((par{i}.sX+2):(par{i}.sX*2+1),k) = par{i}.z;
        par{i}.savX((par{i}.sX*2+3):(par{i}.sX*3+2),k) = par{i}.u;
        
        d.X         = d.X + sum(par{i}.x.^2);
        d.Z         = d.Z + sum(par{i}.z.^2);
        d.U         = d.U + sum(par{i}.u.^2);
        xmz         = (par{i}.x - par{i}.z);
        d.XmZSq     = d.XmZSq + sum(xmz.*xmz);
        zmz         = par{i}.z - par{i}.zOld;
        d.ZmZSq     = d.ZmZSq + sum(zmz.*zmz);
    end 

    saveX(1:(p+1),k) = zglobal;
    if k == 2000
        stophere = 0;
    end
    
    % diagnostics, reporting, termination checks
    history.objval(k)  = d.Fx/n;       % all obj scaled by n in nodes
    history.r_norm(k)  = sqrt(d.XmZSq);
    history.s_norm(k)  = rho*sqrt(d.ZmZSq);

    history.eps_pri(k) = sqrt(n + npar*(p+2))*tol + reltol*max(sqrt(d.X), sqrt(d.Z));
    history.eps_dual(k)= sqrt(n + npar*(p+2))*tol + reltol*rho*sqrt(d.U);

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
saveX(p+n+2,1:k) = history.objval;
fprintf('num iter: %d, fpval: %1.7f, rho: %1.7f\n', k, ...
    history.objval(k), zglobal(1));

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
primal.xi  = zeros(n,1);
omega      = zeros(n,1);
primal.a   = zglobal(2:end);
primal.rho = zglobal(1);

for i = 1:npar
    primal.xi(P{i}) = par{i}.z(1:par{i}.n);
    omega(P{i})     = par{i}.z((par{i}.n+1):par{i}.n*2);
end

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
        gap     = abs(omega + primal.xi);
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