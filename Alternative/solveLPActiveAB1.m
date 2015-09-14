function[out, exitflag, primal, time] = ...
                    solveLPActiveAB1(H, D, options, wrm_strt, NN, ass, dbg, verbose)
% SOLVELPACTIVEAB1 Solve LP with active set strategy
% This function iteratively solves a candidate set of columns (initialised by chooseInitCol)
% for the Restricted Dual RMP, converts to primal variables and adds in any violating columns.
% This uses MATLABs standard linprog solver for the optimisation.

%   Arguments:
%   H          -   (Matrix) inequality constraint corr. to Hu <= beta
%   D          -   (Scalar) upper box constraint
%   options    -   (Options) MATLAB optimisation settings for linprog.
%   wrm_strt   -   (Vector) Index vector of any columns to be immediately
%                  entered into the candidate set.
%   NN         -   (Scalar) Number of violating columns to add in each iter.
%                  either a (%) of size(H,2) if in (0,1) or an absolute number.
%   ass        -   (Scalar) Active set strategy. 
%                  *(-1) Add ALL violating columns in each iter; 
%                  *(0) Add min(NN, num_violations) columns in each iter;
%                  *(0,1] Use Gram-Schmidt strategy where selection is weighted
%                  by violation amt, and ass corresponds to % projection removed 
%                  in each stage; 
%                  *(1,2] same, but not weighted on violation amt;
%                  *(3) choose columns most *similar* in *direction* to most 
%                  violating column.
%   dbg        -   (Logical) Certain debugging options enabled. Checks that active
%                  set strategy identifies same solution as full problem.
%   verbose    -   (Logical) prints summary of activeset optimisation upon exit.
%
%
%   Outputs:
%   out        -   Dual variables - concatenation of (u, beta).
%   exitflag   -   exit flag of final optimisation - currently not used.
%   primal     -   primal variables recovered by solveWeights.
%   time       -   breakdown of time per inner iteration
%

tol = 1e-5;
time = zeros(1, 200);

% Initialise size vars
[dim, N] = size(H);
if NN < 1; NN = ceil(N*NN); end;
if NN == 0; NN = ceil(N/100); end;

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

% ==== DEBUG: SOLVE TOTAL LP (ie without ActiveSet) =======
if dbg > 1
    [f, A, b, Aeq, beq, lb, ub] = createProblem(H, D);
    [orSol, fval , exitflag, ~, lambda] = linprog(f,A,b,Aeq,beq, lb, ub, [], optimset('Display','none'));
    orU    = orSol(1:(end-1))';
    orBeta = orSol(end);
end
% =====================================================

Pnsel = setdiff(1:N, P);
Hnsel = H(:, Pnsel)';
tt = 1;
x = zeros(length(P) + 1 ,1);    % + 1 for beta term

while(true),
    tic;
    
    % Solve restricted (+ warm start)
    [f, A, b, Aeq, beq, lb, ub] = createProblem(Hsel, D);
    [x, ~, exitflag] = linprog(f,A,b,Aeq,beq,lb,ub, x, options);

    % ---- ERROR HANDLING -------------------------------------------------
    % Small values corresponding to low scale kernels can confuse dual-simplex
    remedyme = false;
    if(exitflag ~=1); 
        fprintf('\n optim failed: exitflag = %d ',exitflag);
        remedyme = true;
    elseif sum(x(1:(end-1))) < 1 - tol
        fprintf('\n u does not sum to 1');
        remedyme = true;
    end
    if remedyme
        exitflag = 0;
        optimiters = 100;
        while exitflag == 0
            [x, ~, exitflag] = linprog(f,A,b,Aeq,beq,lb,ub,x, ...
                    optimset('Display','None','MaxIter',optimiters));
            optimiters = optimiters + 100;
        end
        if (sum(x(1:(end-1))) < 1 - tol) || exitflag ~=1
            fprintf('\n Simplex and Interior Point failed.')
            remedyme = true;
        end
    end
    % ---------------------------------------------------------------------
    
    % Check optimality
    u    = x(1:(end-1))';
    beta = x(end);
    
%      [f, Aeq, beq] = createStandardFormProblem(Hsel,D);
%      x2=augLagOpt3(f, Aeq, beq, [], 200, 1e-9, 1,2);
%     x2=augLagOpt4(f, Aeq, beq, [], 200, 1e-9, 1,1e-6);
%     u = x2(1:size(Hsel,2))';
%     beta = f*x2;
    
    [a, rho, xi, fpval, exitflag] = solveWeights(beta, u, Hsel, D, 'exit');
    % check linprog has given an acceptable solution
    if exitflag == -1
        fprintf('TolCorr')
        optsHighTol = optimoptions(options, 'TolCon', 1e-8, 'TolFun', 1e-10);
        [x, ~, exitflag] = linprog(f,A,b,Aeq,beq,lb,ub, x, optsHighTol);
        u    = x(1:(end-1))';
        beta = x(end);
        [a, rho, xi, fpval, ~] = solveWeights(beta, u, Hsel, D, 'nnls');
    end

    % ==== DEBUG: CHECK PRIMAL, and if R2MP == RMP? =======
    if dbg > 1
        [f, A, b, Aeq, beq, lb, ub] = createPrimalProblem(Hsel, D);
        [orSol, fppval , exitflag, ~, lambda] = linprog(f,A,b,Aeq,beq, ...
            lb, ub, [], optimset('Display','none'));
        orA = orSol(1:dim);
        orXi = orSol((dim+1):(end-1));
        orRho = orSol(end);
        disp([beta, -fpval, fppval])
    end
    % =====================================================
    
    %Check primal feasibility (ie. xi = 0)
    vv = Hnsel*a;
    
    num_violations = sum(vv < (rho - tol));
    if num_violations == 0 || isempty(Pnsel) % No violating column: exit
         % === DEBUG: ERROR IF ACTIVESET STRATEGY FAILED ========
         if dbg > 1 && abs(orBeta - beta)>1e-3
             warning('ActiveSet has different solution: actual %1.4f vs %1.4f', ...
                 -orBeta, -beta);
         end
         % =======================================================
        time(tt) = toc;
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
    P                = [P Pnsel(newCols)];
    Pnsel(newCols)   = [];
    Hsel             = H(:,sort(P));
    Hnsel(newCols,:) = [];       % Hnsel is stored as transpose for H^T a.
    
    % Warm start
    x = [x(1:(end-1)); zeros(length(P) - (size(x,1)-1),1); x(end)];
    
    % admin
    NN = min(floor(NN),length(Pnsel));
    time(tt) = toc;
    tt = tt+1;
end

% Set output
out      = zeros(N+1, 1);
out(sort(P))   = u;
out(end) = beta;
primal.a = a;
primal.rho = rho;
primal.xi = zeros(1,N);
primal.xi(sort(P)) = xi;
primal.fpval = fpval;
primal.i = P;
time = time(1:tt);


%fprintf(1, repmat('\b',1, l_msg));
if verbose
    fprintf('R2MP iter = %d, cols_sel = %d, nonzero weight = %d, Primal: ', ...
            tt, length(P), sum(u>tol));
end
end

% function I = chooseInitCol(H, NN)
% % DMR play-around - choosing cols that are most negative
% % - sensible idea but grossly inefficient. See below
% dim = size(H, 1);
% cc = -1000000*ones(dim, 1);
% 
% msquare = @(x, y) (x-y).^2;
% [~, pos] = sort(sum(bsxfun(msquare, cc, H), 1));
% I = pos(1:NN);
% end

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