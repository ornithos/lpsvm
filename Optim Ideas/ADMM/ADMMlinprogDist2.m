function [z, history] = linprogDist2(c, A, b, ignore, rho, npar, MAX_ITER)
% linprog  Solve standard form LP via ADMM
%
% [x, history] = linprog(c, A, b, rho, alpha);
%
% GENERAL CONSENSUS FORM
% Solves the following problem via ADMM:
%
%   minimize     c'*x
%   subject to   Ax = b, x >= 0
%
% The solution is returned in the vector x.
%
% history is a structure that contains the objective value, the primal and
% dual residual norms, and the tolerances for the primal and dual residual
% norms at each iteration.
%
% rho is the augmented Lagrangian parameter.
%
% ignore = (Scalar) ignore first n parameters in consensus.
%
% More information can be found in the paper linked at:
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
%

t_start = tic;

QUIET    = 0;
ABSTOL   = 1e-4;
RELTOL   = 6e-3;

if iscell(A)
    if ~(iscell(b) && iscell(c))
        error('both b and c must be cells also')
    end
    Apar = A;
    bpar = b;
    cpar = c;
    npar = length(A);
    [~, n] = size(A{1});
else
    error('A must be a cell containing the partition of the constraint')
end

% initialise vars
z = zeros(n-ignore, npar);
xFull = zeros(n, npar);
x = zeros(n-ignore, npar);
u = zeros(n-ignore, npar);
F = eye(n);
F(1:ignore, 1:ignore) = zeros(ignore);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

for k = 1:MAX_ITER

    % x-update
    for i = 1:npar
        % NOTE THAT QUADPROG INTERNALLY MULTIPLIES THE FIRST ARGUMENT BY 0.5!!!!
        [xFull(:,i),~,ef] = quadprog(F, cpar{i} - [zeros(ignore,1); z(:,i) - u(:,i)], ...
            [], [], Apar{i}, bpar{i}, zeros(n,1), [], [], optimset('Display','None'));
        
        if ef ~=1
            stophere = 1;
        end
        x(:,i) = xFull((ignore+1):end,i);
    end
    
    % z/u-update
    zold = z;
    z = mean(x,2)*ones(1,npar);
    u = u + x - z;

    % diagnostics, reporting, termination checks
    v = 0;
    for i = 1:npar; v = v + cpar{i}'*xFull(:,i); end
    history.objval(k)  = v/npar;

    history.r_norm(k)  = norm(x - z);
    history.s_norm(k)  = norm(-rho*(z - zold));

    history.eps_pri(k) = sqrt(n*npar)*ABSTOL + RELTOL*max(norm(x), norm(-z));
    history.eps_dual(k)= sqrt(n*npar)*ABSTOL + RELTOL*norm(rho*u);

    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end
end

if ~QUIET
    toc(t_start);
end
z = z(:,1);
end




% Partition the data into npar blocks.
% DIFFERENT TO LPSVM partitionData: PARTITIONS BY ROWS NOT COLUMNS
function [P, pns, Hout] = partitionData(H, npar)
n     = size(H,1);
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
    Hout{i} = H(P{i},:);
end
end