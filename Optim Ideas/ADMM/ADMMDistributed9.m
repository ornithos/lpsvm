function [primal, fval, exitflag, zu] = ADMMDistributed9(H, D, rho, alpha, ...
                        maxIter, tol, reltol, npar, x0, search_typ, verbose)
% ADMMDistributed: Solve LP via ADMM using distributed version of
%   Boyd & Parikh chapter 5. The idea is to solve f^T x s.t. A*x = b
%   and sum(x) = 1 according to an unconstrained linear system. The linear
%   system can be solved in parallel for each partition of data, and the
%   non-negativity and averaging is computed in a centralised step.
%
%   v9
%   -- *** Golden Child Version***
%   -- Line Search Implemented
%   -- Cholesky Factorisation Implemented 
%   -- (DEPRECATED) Return u/z and accept warm start for precision rpt.
%   -- Removed solve dual option - stand-alone implementation is better.
%   -- Removed diagnostics if user supplies [] empty tolerance (speedup)
%   -- solving x via ADMM (KKT) iterations.
%   -- Primal version of ADMM using consensus across nodes.
%   -- Excluding xi from consensus entirely. 
%   -- 'Parallel' element computed in serial.
%
%   Arguments:
%   H          -   (Matrix) inequality constraint corr. to Hu <= beta
%   D          -   (Scalar) upper box constraint
%   rho        -   (Scalar) quadratic penalty constant in augm. lagrangian
%   alpha      -   (Scalar) Under-relaxation parameter in (0,1].
%   maxIter    -   (Scalar) maximum number of iterations in ADMM loop
%   tol        -   (Scalar) stopping criterion.
%   reltol     -   (Scalar) relative convergence component.
%   npar       -   (Scalar) Number of partitions / nodes to simulate.
%   search_typ -   *(false) - return only most recent iterate.
%                  *(true)  - perform line search on iterates.
%                  *(2)     - use the best objective value.
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
if ~isempty(x0)
    if ~(isfield(x0, 'z') && isfield(x0, 'u'))
        error('if x0 is supplied, it must be a structure containing x and u');
    end
end

[p, n]        = size(H);         % n is total variables across partitions
par           = cell(npar,1);    % Parallel parameters stored in par
D2            = n*D;             % rescale D
rho           = n*rho;           % we rescale the objective by n, so rho too.
bDiagn        = ~isempty(tol);   % if no tol supplied, switch off diagnostics

%% Partition data
[P, np, Hpar] = partitionData(H, npar);

for i = 1:npar
% create arrays for f(x): Least Squares
% note that f is scaled by n
    par{i}.n    = np(i);
    par{i}.H    = Hpar{i};
    par{i}.idx  = P{i};
    par{i}.f    = [D2*ones(np(i),1); zeros(np(i),1); -par{i}.n; zeros(p,1)]; 
    par{i}.A    = [eye(np(i)), -eye(np(i)), -ones(np(i),1), par{i}.H';
                    zeros(1,2*np(i)+1), ones(1,p)];
    par{i}.b    = [zeros(np(i),1); 1];
    par{i}.sX   = np(i)*2+p+1;
    
    par{i}.M     = [rho*eye(par{i}.sX), par{i}.A'; par{i}.A, zeros(np(i)+1)];
    par{i}.cholM = chol(par{i}.M*par{i}.M)';
    par{i}.bRHS  = [(-rho*par{i}.f + [zeros(2*par{i}.n +1,1); ones(p,1)]); 
                    -(D2+par{i}.n)*ones(par{i}.n,1); 0];
    
    par{i}.z    = [zeros(1,1); zeros(par{i}.n*2, 1); ones(p,1)./p];
    par{i}.zOld = [zeros(1,1); zeros(par{i}.n*2, 1); ones(p,1)./p];
    par{i}.u    = zeros(par{i}.sX, 1);
    if bDiagn
        par{i}.saveX = zeros(p+1, maxIter);
    end
end

saveX          = zeros(p+1+1,maxIter);

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

% Global Variables
gl_X        = zeros(p+1,npar);
gl_U        = zeros(p+1,npar);

% Warm start if applicable
if ~isempty(x0)
    for i=1:npar
        gl_U(:,i) = x0.u{i}((end-p):end);
        par{i}.u = x0.u{i};
        par{i}.z = x0.z{i};
    end
end

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
        RHS = par{i}.bRHS + addlRHS(rho, par{i}.z, par{i}.u, par{i}.H, par{i}.n, p);
        % -- Solve linear system using Cholesky -----
        w   = linsolve(par{i}.cholM,RHS,opts);
        tmp = linsolve(par{i}.cholM,w,optsT);
        % -------------------------------------------

        par{i}.x    = tmp(1:par{i}.sX);
        par{i}.xHat = alpha.*par{i}.x + (1-alpha).*par{i}.z;
        gl_X(:,i)   = par{i}.xHat((end-p):end);
    end

    % GATHER: z-update (centralised subset, (a, rho))
    gl_Z        = mean(gl_X + gl_U, 2);
    gl_Z(2:end) = max(gl_Z(2:end),0);   % rho does not have to be positive.
    
    for i = 1:npar
        par{i}.zOld  = par{i}.z;
        
        par{i}.z  = [max(par{i}.xHat(1:(end-p-1))+par{i}.u(1:(end-p-1)), 0);
                    gl_Z];
        par{i}.u  = par{i}.u - par{i}.z + par{i}.xHat;
        gl_U(:,i) = par{i}.u((end-p):end);
    end
    
    %% Diagnostics
    saveX(1:(p+1),k) = gl_Z;
    if bDiagn
        % >>>> Perform full diagnostics and termination checks
        for i = 1:npar
            d.X         = d.X + sum(par{i}.x((par{i}.n+1):end).^2);
            d.Z         = d.Z + sum(gl_Z.^2);
            d.U         = d.U + sum(par{i}.u.^2);
            xmz         = (par{i}.x - (par{i}.z));
            d.XmZSq     = d.XmZSq + sum(xmz.*xmz);
            zmz         = par{i}.z - par{i}.zOld;
            d.ZmZSq     = d.ZmZSq + sum(zmz.*zmz);
            
            par{i}.saveX(:,k) =  par{i}.x((end-p):end);
        end 

        % diagnostics, reporting, termination checks
        history.objval(k)  = D*sum(max(0,gl_Z(1)-H'*gl_Z(2:end)))-gl_Z(1);
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
    elseif search_typ == 2
        % >>>> Pootle along faster in complete ignorance.
        % -- use best objective value --
        history.objval(k)  = D*sum(max(0,gl_Z(1)-H'*gl_Z(2:end)))-gl_Z(1);
    else
        % >>>> Pootle along faster in complete ignorance.
        % -- line search --
        % (still need s_norm for line search)
        for i = 1:npar
            zmz         = par{i}.z - par{i}.zOld;
            d.ZmZSq     = d.ZmZSq + sum(zmz.*zmz);
        end 
        history.s_norm(k)  = rho*sqrt(d.ZmZSq);
    end
end

%% Output
if bDiagn; saveX(end,1:k) = history.objval; end;

fval          = D*sum(max(0,gl_Z(1)-H'*gl_Z(2:end)))-gl_Z(1); % recalc due to xi
exitflag      = 1;
ignoreLSVals  = true;   % for what follows (line search vals)

%% Best Objective or Line Search if oscillating or user request:

if search_typ == 2
    % best objective value (if requested)
    [fval2,bst] = min(history.objval(1:k));
    gl_Z    = saveX(1:(p+1),bst);
    ignoreLSVals = false;        % include existing val search in this
    
    lsval      = fval2 - fval;
    primal.a   = gl_Z(2:end);
    primal.rho = gl_Z(1);
    primal.xi  = max(primal.rho - H'*primal.a,0);
    fprintf('evs: ');
end
if search_typ == 1
    if k == maxIter; exitflag = 0; end;
    % --- poor man's basis pursuit --------------
    
    % OBTAIN LINE - (1)
    % Search most recent oscillation - preferable since trust recent best.
    if k > 400
        
        % set up search region
        bp_rng        = min(k*2/3,500);
        bp_rng        = (k-bp_rng):(k-1);
        bp_diff       = diff(history.s_norm);
        bp_diff_bool  = bp_diff > 0;   % attempting to make more efficient
        ptr           = length(bp_diff)-1;
        prv           = bp_diff_bool(ptr+1);
        found_end     = false;
        
        % find start and end of most recent oscillation
        while 1
            if xor(prv, bp_diff_bool(ptr))
                if ~found_end
                    ts_end = mean(saveX(2:(p+1),(ptr-5):min(ptr+5,end)),2);
                    found_end = true;
                    found_type = prv;
                    ptr = ptr - ceil(length(bp_rng)/10);
                else
                    if found_type ~= prv   % ensuring we get a peak/trough pair
                        ts_bgn = mean(saveX(2:(p+1),(ptr-5):(ptr+5)),2);
                        break
                    end
                end
            end
            if ptr <= bp_rng(1)
                exitflag = -1;
                break
            end
            prv = bp_diff_bool(ptr);
            ptr = ptr-1;
        end
    end
    
    % OBTAIN LINE - (2)
    % Search most recent 50 iterations
    if ~(search_typ && k > 400) || exitflag == -1;
        bp_rng        = round(max(5,min(k*1/5,50)));  % between [5,50] iterates
        bp_rng        = (k-bp_rng+1):k;
        bp_raw        = history.s_norm(bp_rng);
        [~,bp_sorted] = sort(bp_raw,'ascend');
        cols          = bp_sorted(1:floor(length(bp_raw)/2));
        ts_bgn        = mean(saveX(2:(p+1),k-cols+1),2);
        cols          = bp_sorted(floor(1+length(bp_raw)/2):end);
        ts_end        = mean(saveX(2:(p+1),k-cols+1),2);
    end
    
    % perform line search if evidence of non-convergence.
    if norm(ts_bgn - ts_end) > 1e-4
        [ts_bgn, ts_end]  = lsBoundarySimplex(ts_bgn, ts_end);

        [primal.a, primal.rho, primal.xi, fval2] = ...
                solve2DBisection2(H, D, ts_bgn,ts_end);
        
        if fval2 <= fval
            ignoreLSVals = false;
            lsval        = fval2 - fval;
            fval         = fval2;
        end
    end
 end

% Unless line search successful, we use these 'converged' values.
if ignoreLSVals
    primal.a   = gl_Z(2:end);
    primal.rho = gl_Z(1);
    primal.xi  = max(primal.rho - H'*primal.a,0);
    lsval      = 0;
end
fprintf('num iter: %d, fpval: %1.7f, rho: %1.7f, ls: %1.7f', k, ...
    fval, primal.rho, lsval);

% can return copies of variables if requested
if nargout == 4
    zu.z = cell(npar,1);
    zu.u = cell(npar,1);
    for i=1:npar
        zu.z{i} = par{i}.z;
        zu.z{i}((end-p):end) = [primal.a; primal.rho];
        zu.u{i} = par{i}.u;
    end
end
end


% Additional (Variable) part of RHS term updated with new (z, u)
% Note that changing rho requires much more than this
function out = addlRHS(rho, z, u, H, n, p)
zmu = z - u;
a   = zmu((end-p+1):end);
HTa = H'*a;
out = [rho^2.*zmu;
       rho.*(zmu(1:n) - zmu((n+1):(2*n)) - zmu(2*n+1)*ones(n,1) + HTa);
       rho.*sum(a)];
end


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