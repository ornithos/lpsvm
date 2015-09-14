function [primal, dual, fval, exitflag] = ADMMDistributed4(H, D, rho, eta, ...
                        maxIter, tol, reltol, x0, npar, dualreq, verbose)
% ADMMDistributed: Solve LP via ADMM using distributed version of
%   Boyd & Parikh chapter 5. The idea is to solve f^T x s.t. A*x = b
%   and sum(x) = 1 according to an unconstrained linear system. The linear
%   system can be solved in parallel for each partition of data, and the
%   non-negativity and averaging is computed in a centralised step.
%
%   v4
%   -- Nesterov Acceleration
%     -- This doesn't help much. The issue is that we have a flat gradient
%     while the control signal (u) builds to a critical point, and then
%     change happens fast anyway. Nevertheless, it does speed up the
%     changepoints. They simply aren't the major bottleneck.
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

%% Create Partitions of the data
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
    par{i}.savX = NaN(par{i}.sX+2+2*p+2,maxIter);
    
    par{i}.F    = eye(par{i}.sX);
    par{i}.F(1:np(i),1:np(i)) = zeros(np(i));
end

% Global (Consensus) variables
gl_X        = zeros(p+1,npar);
gl_UHat     = zeros(p+1,npar);
gl_UOld     = zeros(p+1,npar);
gl_Z        = [0; ones(p,1)./p];
gl_ZHat     = [0; ones(p,1)./p];
gl_ZOld     = [0; ones(p,1)./p];

% Nesterov Variables
c              = [0, Inf];              % initialise 'restart' tracker c.
alpha          = [1, 1];
restarts       = 0;

saveX          = zeros(p+1+n+1+npar+1,maxIter);

%% Main ADMM Section
% code heavily adapted from S. Boyd, N. Parikh, E. Chu, B. Peleato, 
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
    
    %% Vanilla Updates
    % 'PARALLEL' x-update
    for i = 1:npar
        % Update x 
        % NOTE THAT QUADPROG INTERNALLY MULTIPLIES THE FIRST ARGUMENT BY 0.5!!!!
        [par{i}.x,~,ef] = quadprog(rho*par{i}.F, par{i}.f + rho*...
            [zeros(par{i}.n,1); (gl_UHat(:,i) - gl_ZHat)], par{i}.A, ...
            par{i}.b, par{i}.Aeq, par{i}.beq, par{i}.lb, [], [], ...
            optimset('Display','None')); 
        if ef ~= 1
            stophere = 1
        end
        gl_X(:,i) = par{i}.x((par{i}.n+1):end);
    end

    % GATHER: z-update (a/rho) = MEAN(x_{global})
    % average(u)=0 for each global var after 1st iter (and = 0 in 1st iter).
    % TEMPORARY CHECK IF sum uHat == 0. THIS (SHOULD BE) A RESULT.
    if norm(mean(gl_UHat,2)) > 1e-5
        stophere = 1
    end
    gl_Z    = mean(gl_X,2);   % avg (x+u)
    
    
    % 'PARALLEL' u-update and local subset of z (xi, w).
    gl_U = gl_UHat + gl_X - repmat(gl_Z,1,npar);
    
    %% Nesterov Accleration
    uMu  = gl_U - gl_UHat;
    zMz  = gl_ZHat - gl_Z;   % opp sign due to matrix B. But redundant.
    c(1) = sum(sum(uMu.*uMu)) + sum(sum(zMz.*zMz));
    
    % main Nesterov Branch (create accelerated estimates if not overshooting)
    if c(1) < eta*c(2)
        alpha(1) = (1 + sqrt(1 + 4*alpha(2)^2))/2;
        a        = (alpha(2)-1)/alpha(1);
        gl_ZHat  = (1 + a)*gl_Z -a*gl_ZOld;
        gl_UHat  = (1 + a)*gl_U -a*gl_UOld;
    else
        alpha(1) = 1;
        gl_ZHat  = gl_Z;        % Changed/I think this is a misprint in Goldstein.
        gl_UHat  = gl_U;
        c(1)     = c(2)/eta;
        restarts = restarts + 1;
    end
    
    % Track z for debugging purposes
    saveX(1:(p+1),k) = gl_Z;
    
    %% >>>> for diagnostics
    for i = 1:npar
        saveX((p+1)+P{i},k) = par{i}.x(1:par{i}.n);
        d.Fx        = d.Fx + par{i}.f'*par{i}.x;
        
        saveX(p+n+2+i,k) = par{i}.f'*par{i}.x;
        par{i}.savX(1:par{i}.sX,k) = par{i}.x;
        par{i}.savX((par{i}.sX+2):(par{i}.sX+2+p),k) = gl_Z;
        par{i}.savX((par{i}.sX+2+p+2):(par{i}.sX+2+2*p+2),k) = gl_U(:,i);
    end 
    
    d.X         = sum(sum(gl_X.^2));
    d.Z         = sum(gl_Z.^2);
    d.U         = sum(sum(gl_U.^2));
    xmz         = gl_X - repmat(gl_Z, 1, npar);
    d.XmZSq     = sum(sum(xmz.*xmz));
        
    
    % Update history
    history.objval(k)  = d.Fx/n;       % all obj scaled by n in nodes
    history.r_norm(k)  = sqrt(d.XmZSq);
    history.s_norm(k)  = rho*norm(gl_Z - gl_ZOld);

    history.eps_pri(k) = sqrt(p+1)*tol + reltol*max(sqrt(d.X), sqrt(d.Z));
    history.eps_dual(k)= sqrt(p+1)*tol + reltol*rho*sqrt(d.U);

    if verbose
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

    % Check for convergence
    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end
    
    
    % Update Nesterov Variables for next iteration
    c(2) = c(1);
    alpha(2) = alpha(1);
    gl_ZOld = gl_Z;
    gl_UOld = gl_U;
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