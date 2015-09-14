function[out, exitflag, primal, time, msgl] = solveLPActiveAD1(H, D, options, ...
                  wrm_strt, NN, ass, npar, npiter, tol, dbg, verbose)
% solveLPActiveAD1: 
% ----------------- Solve LP via active set identification. Unlike
%   solveLPActiveAB1, this method uses the (serial) ADMM optimization,
%   which adds approximation to the solution. And so more care must
%   be taken with active set identification. Hopefully we will move the
%   ADMM Optimization to a more efficient implementation yielding better
%   solutions (for instance, feasible ones..).
%
%   v1.
%
%   Arguments:
%   H          -   (Matrix) inequality constraint corr. to Hu <= beta
%   D          -   (Scalar) upper box constraint.
%   options    -   (--unused--) options for optimization.
%   wrm_start  -   (Vector) Optional. Indices of warm start active set.
%   NN         -   (Scalar) Optional. Number of columns to add to the
%                  active set on each iteration. Can be absolute number or
%                  fraction of columns in H (if NN < 1).
%   ass        -   (Scalar) Active set strategy. Most of these are duds, so
%                  no need to use this.
%   npar       -   (Scalar) Number parallel 'cores'
%   npiter     -   (Scalar) Number parallel iterations in active set solver
%   tol        -   (Scalar) Tolerance for optimization.
%   dbg        -   (Logical) Run checks against non-active set LP.
%   verbose    -   (Logical) Feedback on number of active set iterations.

time   = zeros(1, 200);

% Initialise size vars
[dim, N] = size(H);
if NN < 1; NN = ceil(N*NN); end;
if NN == 0; NN = ceil(N/100); end;

%% Create Initial Active Set/
if ceil(1/D) + dim < N
    num_needed = ceil(1/D) + dim;
    
    if ~isempty(wrm_strt)
        % include warm start columns if available
        P = wrm_strt;
        num_needed = num_needed - length(P);
        if num_needed > 0
            remaining = setdiff(1:N,P);
            Hsel = H(:,remaining);
            % SHOULD THINK ABOUT MAKING THESE NEXT CHOICES ORTHOGONAL!
            P = union(P, remaining(chooseInitCol(Hsel, num_needed)));
        end
    else
        % warm start unavailable
        P = chooseInitCol2(H, ceil(1/D) + dim, D);
    end
    Hsel = H(:,P);
else
    % constraints enforce all columns in active set.
    P = 1:size(H,2);
    Hsel = H;
end

%% 

% ==== DEBUG: SOLVE TOTAL LP (ie without ActiveSet) =======
if dbg > 1
    [f, A, b, Aeq, beq, lb, ub] = createProblem(H, D);
    [orSol, fval , exitflag, ~, lambda] = linprog(f,A,b,Aeq,beq, lb, ub, [], optimset('Display','none'));
    orU    = orSol(1:(end-1))';
    orBeta = orSol(end);
end
% =====================================================

% Admin
Pnsel = setdiff(1:N, P);
tt = 1;
x = zeros(length(P) + 1 ,1);    % + 1 for beta term


%% Main Loop
while(true),
    ticAD = tic;
    
    % Solve restricted (+ warm start)
%     [f, A, b, Aeq, beq, lb, ub] = createProblem(Hsel, D);
%     [x, ~, exitflag] = linprog(f,A,b,Aeq,beq,lb,ub, x, options);
if dim == 2
    [primal.a, primal.rho, primal.xi, fpval] = ...
        solve2DBisection2(Hsel, D, [0;1],[1;0]);
    exitflag = 1;
else
    step = max(0.5,log(length(P)/4));
%     step = max(0.5,2*floor(nthroot(length(P),3)));
%     [z, history] = ADMMDirect3(Hsel, D, step, 1, 3000, ...
%                                 tol, tol*10, false);
%     [primal, dual, fpval, exitflag] = ADMMDirect4(Hsel, D, step, 0.9999, npiter, ...
%                                 tol/10e3, tol/10e3, [], false, false);      %% Serial NNLS: Chol
    [primal, fpval, exitflag] = ADMMDistributed9(Hsel, D, step, 1.5, npiter,...
                            tol/10, tol/20, npar, [], true, false); %% NNLS: Chol
%     [primal, fpval, exitflag, zu0] = ADMMDistributed9(Hsel, D, step, 1.5, ...
%                 npiter, [], [], npar, [], true, false); %% NNLS: No Diagnostics
%     [primal, dual, fpval, exitflag] = ADMMDistributed3(Hsel, D, 3, 0.9999, 3000, ...
%                                 tol*10, tol*20, [], 4, false, true); %% QP
%     [primal, dual, fpval, exitflag] = ADMMDistributed7(Hsel, D, 3, 1.6, 3000, ...
%                                 tol*10, tol*20, [], 4, false, true); %% NNLS Relax: Chol
%       [primal, dual, fpval, exitflag] = ADMMHinge1(Hsel, D, 1, 1, 2000, ...
%                                 tol*10, tol*20, [], 4, false, false); %% Prox
end
    % ---- ERROR HANDLING -------------------------------------------------
    % Small values corresponding to low scale kernels can confuse dual-simplex
%     remedyme = false;
%     if(exitflag ~=1); 
%         fprintf('\n optim failed: exitflag = %d ',exitflag);
%         remedyme = true;
%     elseif sum(x(1:(end-1))) < 1 - tol
%         fprintf('\n u does not sum to 1');
%         remedyme = true;
%     end
%     if remedyme
%         exitflag = 0;
%         optimiters = 100;
%         while exitflag == 0
%             [x, ~, exitflag] = linprog(f,A,b,Aeq,beq,lb,ub,x, ...
%                     optimset('Display','None','MaxIter',optimiters));
%             optimiters = optimiters + 100;
%         end
%         if (sum(x(1:(end-1))) < 1 - tol) || exitflag ~=1
%             fprintf('\n Simplex and Interior Point failed.')
%             remedyme = true;
%         end
%     end
    % ---------------------------------------------------------------------
    
    % Check optimality
    a    = primal.a;
    rho  = primal.rho;
    xi   = primal.xi;

%     display((Hsel*u - beta)');
    % ==== DEBUG: CHECK PRIMAL, and if R2MP == RMP? =======
    if false %dbg > 1
        [f, A, b, Aeq, beq, lb, ub] = createPrimalProblem(Hsel, D);
        [orSol, fppval , exitflag, ~, lambda] = linprog(f,A,b,Aeq,beq, ...
            lb, ub, [], optimset('Display','none'));
        orA = orSol(1:dim);
        orXi = orSol((dim+1):(end-1));
        orRho = orSol(end);
        fprintf(' opt: %1.7f\n', fppval);
        %disp([beta, -fpval, fppval])
    else
        fprintf('\n');
    end
    % =====================================================
    
    %Check primal feasibility (ie. xi = 0)
    Hnsel = H(:, Pnsel)';
    vv = Hnsel*a;
    
    num_violations = sum(vv < (rho - tol));
    if num_violations == 0 || isempty(Pnsel) % No violating column: exit
         % === DEBUG: ERROR IF ACTIVESET STRATEGY FAILED ========
         if dbg > 1 && abs(orBeta - beta)>1e-3
             warning('ActiveSet has different solution: actual %1.4f vs %1.4f', ...
                 -orBeta, -beta);
         end
         % =======================================================
        dual = solveWeightsDual(Hsel, D, primal.a, primal.rho, primal.xi, fpval);
        time(tt) = toc(ticAD);
        break;
    end
    
    % Add violating columns
    [vv,ss]         = sort(vv);
    searchMax       = min(NN*100, num_violations);
    numAddl         = min(NN, num_violations);
    if ass == -1
        newCols = ss(1:num_violations);
    elseif ass == 0
        newCols = ss(1:numAddl);
    elseif ass <= 2
        ss              = ss(1:searchMax); % search only the first searchMax violations
        vv              = vv(1:searchMax);
        if ass <= 1
            newCols     = pickIndependentCols(H, ss, -vv, numAddl, ass);
        elseif ass <= 2
            newCols     = pickIndependentCols(H, ss, ones(size(vv)), numAddl, ass-1);
        end
    elseif ass ==3
        searchMax = min(length(Pnsel), numAddl*5);
        mvc = ss(1);        % most violating column
        nextviolations = ss(2:searchMax);
        Nvc = H(:,Pnsel(nextviolations));  % (N)ext most violating columns
        Nvc = (H(:,Pnsel(mvc))'*Nvc)./sqrt(sum(Nvc.^2));
        [~,ss2] = sort(Nvc, 'descend');
        newCols = ss2(1:numAddl);
        newCols = [mvc; nextviolations(newCols)];
    else
        error('Invalid activeset strategy selected.');
    end
    P               = [P Pnsel(newCols)];
    Pnsel(newCols)   = [];
    Hsel            = H(:,P);
    
    % admin
    NN = min(floor(NN),length(Pnsel));
    time(tt) = toc(ticAD);
    tt = tt+1;
end

% Set output
out      = zeros(N+1, 1);
out(P)   = dual.u;
out(end) = dual.beta;
primal.a = a;
primal.rho = rho;
primal.xi = zeros(1,N);
primal.xi(P) = xi;
primal.fpval = fpval;
primal.i = P;
time = time(1:tt);


%fprintf(1, repmat('\b',1, l_msg));
if verbose
    msgl = fprintf('R2MP iter = %d, cols_sel = %d, nonzero weight = %d, Primal: ', ...
            tt, length(P), sum(dual.u>tol));
else
    msgl = 0;
end
end

function I = chooseInitCol(H, NN)
% optimised version
[~, pos] = sort(sum(H));
I = pos(1:NN);
end


function I = chooseInitCol2(H, NN, D)

rows = size(H,1);
numfill = floor(1/D);
%remainder = 1 - numfill*D;
solosGt0 = zeros(1,size(H,2));
for cI = 1:rows
    [~,ordered] = sort(H(cI,:));
    solosGt0(ordered(1:numfill)) = ...
        solosGt0(ordered(1:numfill)) + ones(1, numfill);
end

% add col total information to break ties
breakties = -sum(H, 1);
quicksearch = min(100, size(H,2));
breakties = breakties ./ (1000 * max(breakties(1:quicksearch)));

[~,ordered] = sort(solosGt0 + breakties, 'descend');

I = ordered(1:NN);
end


function [f, A, b, Aeq, beq, lb, ub] = createProblem(H, D)
[dim, N] = size(H);

f = zeros(1, N+1); f(end) = 1; % Objective function = beta

A   = [H, -ones(dim,1)]; % Hu <= beta
b   = zeros(dim, 1);
Aeq = [ones(1, N) 0]; % sum u = 1
beq = 1;
lb = [zeros(1, N) -Inf]; % u beta
ub = [D*ones(1, N) Inf]; % u beta

end