function [z, history] = linprogDist3(c, A, b, lb, ignore, rho, npar, MAX_ITER)
% linprog  Solve standard form LP via ADMM
%
% [x, history] = linprog(c, A, b, rho, alpha);
%
% 3. GENERAL CONSENSUS FORM
%
% ignore = (Scalar) ignore first n parameters in consensus.
%           if negative - interpret as keep last (-ignore) params in
%           consensus.
%
% More information can be found in the paper linked at:
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
%
% RHO IS MISSING FROM OPTIMISATION OF X1

t_start = tic;

QUIET    = 0;
% ABSTOL   = 1e-4;
% RELTOL   = 6e-3;
ABSTOL   = 1e-6;
RELTOL   = 1e-4;

if iscell(A)
    if ~(iscell(b) && iscell(c))
        error('both b and c must be cells also')
    end
    Apar = A;
    bpar = b;
    cpar = c;
    npar = length(A);
else
    error('A must be a cell containing the partition of the constraint')
end

% initialise vars
keep    = cell(npar,1);
discard = zeros(npar,1);
xFull   = cell(npar,1);
F       = cell(npar,1);
n       = 0;

if ignore < 0
    for i = 1:npar;
        varn = size(Apar{i}, 2);
        keep{i} = (varn+ignore+1):varn;
        discard(i) = varn+ignore;
        n = n + varn;
    end
else
    for i = 1:npar;
        varn = size(Apar{i},2);
        keep{i} = (ignore+1):varn;
        discard(i) = ignore;
        n = n + varn;
    end
end

for i = 1:npar
    xFull{i} = zeros(length(keep{i})+abs(ignore),1);
    F{i}     = eye(length(keep{i})+discard(i));
    F{i}(1:discard(i), 1:discard(i)) = zeros(discard(i));
end
x   = zeros(length(keep{1}), npar);
z   = zeros(length(keep{1}), npar);
u   = zeros(length(keep{1}), npar);
    
saveX          = zeros(npar*(1+length(keep{1})),MAX_ITER);
if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

for k = 1:MAX_ITER
    
    g = cell(npar,1);
    % x-update
    for i = 1:npar
        % NOTE THAT QUADPROG INTERNALLY MULTIPLIES THE FIRST ARGUMENT BY 0.5!!!!
        [xFull{i},~,ef] = quadprog(rho*F{i}, ...
            cpar{i} - rho*[zeros(discard(i),1); z(:,i) - u(:,i)], ...
            [], [], Apar{i}, bpar{i}, lb{i}, [], [], optimset('Display','None'));
        
        if ef ~=1
            stophere = 1;
        end
        x(:,i) = xFull{i}(keep{i});
        saveX(((i-1)*(1+length(keep{1}))+1):(i*(1+length(keep{1}))-1),k) = x(:,i);
%         g{i}=sprintf('%1.6f ', xFull{i});
%         fprintf('%s\n', g{i});
    end
    
    

    % z/u-update
    zold = z;
    z = mean(x,2)*ones(1,npar);
    u = u + x - z;

    % diagnostics, reporting, termination checks
    v = 0;
    for i = 1:npar; v = v + cpar{i}'*xFull{i}; end
    history.objval(k)  = v/npar;

    history.r_norm(k)  = norm(x - z);
    history.s_norm(k)  = norm(-rho*(z - zold));

    history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
    history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);

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
% function [P, pns, Hout] = partitionData(H, npar)
% n     = size(H,1);
% rp    = randperm(n);                % randomly order elements
% P     = cell(npar,1);
% Hout  = cell(npar,1);
% pns   = [floor(n/npar), mod(n,npar)]; % number of elements in each partition
% pns   = pns(1)*ones(npar,1) + [ones(pns(2),1); zeros(npar-pns(2),1)];
% pns   = pns(randperm(npar));         % add 0 at beginning for ease in loop.
% cns   = [0; cumsum(pns)];            % randomisation is over num elemnts in
%                                      % each partition when mod(n, npar) != 0
% for i = 1:npar
%     P{i}    = rp((cns(i)+1):cns(i+1));      % randomisation here is over el
%     Hout{i} = H(P{i},:);
% end
% end