function [a, rho, xi, fpval, exitflag] = solveWeights(beta, u, H, D, negstrategy, dbg)
% solveWeights:
% Solve KKT system for primal variables given the optimum dual variables.
% There are many ways this can go wrong, particularly for large problems. See
% report for more details.

%   Arguments:
%   beta        -   (Scalar) optimal dual objective / minimax
%   u           -   (Vector) optimal dual variable u
%   H           -   (Matrix) inequality constraint corr. to *R2MP* Hu <= beta
%   D           -   (Scalar) upper box constraint
%   negstrategy -   (Character) What to do if nonnegative primal variables are
%                   returned as negative. See report and switch statement below.
%   dbg         -   (Logical) Outputs if diverts from basic path of small linear
%                   system. This may be due to lack of constraints or negative
%                   values in the solution of the LS system. > Small inline comments.
%
%
%   Outputs:
%   a          -   Primal variable
%   rho        -   Primal variable
%   xi         -   Primal variable
%   fpval      -   Objective value in primal (useful for debugging)
%   exitflag   -   Roughly follows convention of exitflag from linprog
%

if ~exist('negstrategy', 'var')
    negstrategy = 'NNLS';
end
if ~exist('dbg', 'var')
    dbg = false;
end

tol = 1e-7;

u(u<0) = 0;                    % Precision of linprog can result in negative
uIsNonZero = u > tol;          % u (constraint violation) of O(10^-6). OK.

% (xi_i) if u in dual is greater than zero, inequality in primal is binding
uu = u(uIsNonZero);
HH = H(:, uIsNonZero);

% (a_j) if inequality in dual is not binding, the primal variable is 0
% -- difficult to identify binding Hu = beta since precision depends on
% tolerances in linprog and size of problem. 1e-4 may be = 0 in some cases!
% IGNORE TOLERANCE and work under assumption that at least one a_j > 0!
deltA     = abs(HH*uu' - beta);
zz        = iterativeUnion(deltA, 100, 100, 1e-10);

HH = HH(zz, :);
[nAs, dd] = size(HH);

% Find all free support vectors (on boundary) (xi)
if nAs == 1 && mod(1,D) <= tol
    % covers pathology when only one constraint
    % and D divides 1 exactly => no free sv.s - should have exactly one.
    % This can be resolved by obtaining the 'last' s.v. to have moved -> D
    % It is the index corresponding to the largest derivative (min, so want
    % small) within the indices chosen as s.v.s (ie u>0)
    [~ ,orderU] = sort(HH, 'ascend');
    Xizero = 1:length(uu) == orderU(sum(uIsNonZero));
else
    Xizero = abs(uu-D) > tol;
end

nXiZero = sum(Xizero);
% Test if underdetermined system - either add constraint or remove variable
% =========================================================================
if nXiZero < nAs-1
    
    if dbg; fprintf('UDHack'); end;
    delta   = abs(D-uu);
    [~,ord] = sort(delta, 'descend');
    
    % if any actually equal zero - we don't want this.
    if nAs-1 <= length(uu)
        idx = ord(1:(nAs-1));
        ties = delta(idx) == 0;
        if any(ties)
            idx(ties)      = [];
            nt             = sum(ties);
            [~, deltA_ord] = sort(deltA);
            % remove equivalent number from number of a_j > 0.
            % This is *usually* justified, but occasionally may be flawed.
            rmA     = deltA_ord((nAs - nt + 1):nAs);
            nxtA    = deltA_ord(nAs - nt);
            if dbg
                if deltA(nxtA) > 0.5*deltA(rmA(1))
                    fprintf('_poor correction_')  % probably flawed correction!
                end
            end
            zz(rmA) = false;
            HH = H(zz, uIsNonZero);
            [nAs, dd] = size(HH);
        end
    else
        % when there simply aren't enough non-zero u_i to deal with nA
        lu  = length(uu);
        idx = 1:lu;
        extra = (nAs-1) - lu;
        [~, deltA_ord] = sort(deltA);
        rmA     = deltA_ord((nAs - extra +1):nAs);
        nxtA    = deltA_ord(nAs - extra);
        if dbg
            if deltA(nxtA) > 0.5*deltA(rmA(1))
                fprintf('_poor correction_')  % probably flawed correction!
            end
        end
        zz(rmA) = false;
        HH = H(zz, uIsNonZero);
        [nAs, dd] = size(HH);
    end
    Xizero = false(1,length(uu));
    Xizero(idx) = true;
end
% =========================================================================
% elseif nXiZero > nAs
%     fprintf('ODHack');
%     HH      = H(:, uIsNonZero);
%     zz      = abs(HH*uu' - beta);
%     [~,ord] = sort(zz, 'ascend');
%     HH = HH(ord(1:nXiZero), :);
%     [nAs, dd] = size(HH);
%     zz = zz < tol;
% end

%% LINEAR SYSTEM:

% if have enough constraints can solve small system, otherwise solve larger
if nXiZero <= nAs-1
    % Remove all free s.v.s from system since we already know xi_i = 0.
    xiRes = eye(dd);
    xiRes(:,Xizero) = [];

    rhoRes = -ones(dd, 1);
    A          = [HH' xiRes rhoRes];
    rowLen     = size(A, 2);
    % Constraint on ||a|| = 1 (L1)
    % Need to enforce xi >= 0 and ai >= 0, but unable to in OLS.
    simplexRes = [ones(1, nAs) zeros(1, rowLen-nAs)]; % Sum a = 1
    duality = [zeros(1, nAs), -D*ones(1, rowLen-nAs-1), 1]; % duality gap = 0

    A = [A; simplexRes; duality];
    beq = [zeros(dd, 1); 1; beta];
    if dbg; fprintf('expensiveLS'); end;
    cheap = false;
else
    A = [HH(:, Xizero)', -ones(sum(Xizero),1); [ones(1, sum(zz)), 0]];
    beq = [zeros(sum(Xizero),1); 1];
    cheap = true;
end

% =========================================================
% ==== HACK TO CATCH WARNINGS =============================
s = warning('error','MATLAB:nearlySingularMatrix');
warning('error','MATLAB:rankDeficientMatrix');

% solve system
err = false;
try
    x = A\beq;
catch
    err = true;
    fprintf('mld');
end
warning(s);
% =========================================================

% [a, xi, rho, fpval] = extractPrimalVars(x, H, uIsNonZero, Xizero, nAs, zz, D);
% 
% tst = [H(zz, Xizero)', -ones(sum(Xizero),1); [ones(1, sum(zz)), 1]]\[zeros(sum(Xizero),1); 1];
% if norm(tst - [a; rho]) > 1e-5
%     stophere = 1;
% end

if err || (sum(x(1:(end-1)) < -tol) > size(H,2)/100 && min(x(1:(end-1))) < -1e-5)
    
    % If just return error, then no need to print msg
    if strcmpi(negstrategy, 'EXIT')
        exitflag = -1;
        a = [];
        xi = [];
        rho = [];
        fpval = 0;
        if dbg; fprintf('exitSolve'); end
        fprintf('exitSolve');
        return
    end
    
    fprintf('KKT problem (neg xi/a). Should not occur anymore');

    % Least Squares solution violates nonnegativity constraints.
    if strcmpi(negstrategy, 'SVD')
        [U, S, V] = svd(A);
        m = dd+1;
        if m >= rowLen
            fprintf('Negstrat: SVD... System is not underdetermined!');
            pause;
        end
        
        free = (m+1):rowLen;
        xbase = zeros(rowLen,1);
        for i = 1:m
            xbase = xbase + U(end,i).*V(:,i)./S(i,i);
        end
        x = xbase + 2*V(:,free)*ones(length(free),1);
        % THIS IS THOROUGHLY INCOMPLETE - HAVEN'T MOVED IN THE NULL SPACE.
        [a, xi, rho, fpval] = extractPrimalVars(x, H, uIsNonZero, Xizero, nAs, zz, D);
        
    elseif strcmpi(negstrategy, 'QP')
        % Enforce via QP

        fprintf('Q');
        Hess = [eye(dd) zeros(dd, rowLen); zeros(rowLen, dd + rowLen)];
        f = zeros(dd+rowLen,1);
        Aeq = [[eye(dd); zeros(1,dd)] A];
        lb = [-Inf(1, dd) zeros(1, rowLen-1) -Inf];
        ub = Inf(1, dd+rowLen);
        options =  optimset('Display','none');
        [x, ~, exitflag] = quadprog(Hess, [], [], [], Aeq, beq, lb, ub, [], options);
        if exitflag ~= 1; fprintf('Bad QP exit %d, ', exitflag); end;
        x = x((dd+1):end);
        fprintf('P');
        [a, xi, rho, fpval] = extractPrimalVars(x, H, uIsNonZero, Xizero, nAs, zz, D);
        return
        
    elseif strcmpi(negstrategy, 'NNLS')
        % Enforce by Non-negative Least Squares
        
        fprintf('NN');
        x1 = lsqnonneg(A,beq);
        [a, xi, rho, fpval] = extractPrimalVars(x1, H, uIsNonZero, ...
            Xizero, nAs, zz, D, 1);
        fprintf('L');
        A(:, end) = -A(:,end);
        x2 = lsqnonneg(A,beq);
        [a2, xi2, rho2, fpval2] = extractPrimalVars(x2, H, uIsNonZero, ...
            Xizero, nAs, zz, D, -1);
        fprintf('S');
        
        use2 = false;
        if isinf(fpval) && isinf(fpval2)  % If both objectives return infeasible
            if (sum(a)-1)^2 > (sum(a2)-1)^2  % use closest to feasible
                use2 = true;
            end
        elseif fpval2 < fpval       % Otherwise choose the smaller objective val
                use2 = true;      % (Primal objective here is minimisation)
        end
        if use2
            a = a2;
            xi = xi2;
            rho = rho2;
            fpval = fpval2;
            fprintf(':2 ');
        end
        exitflag = 0;
        return
    elseif ~strcmpi(negstrategy, 'NONE')
        error(['Invalid negative strategy selected: choose from ''QP'', ', ...
            'and ''NNLS''. Option 2 is preferred.']);
    end
end


[a, xi, rho, fpval] = extractPrimalVars(x, H, uIsNonZero, Xizero, nAs, zz, D, 1, cheap);
% if cheap
%     A     = [HH' xiRes rhoRes];
%     A     = [A; simplexRes; duality];
%     beq   = [zeros(dd, 1); 1; beta];
%     x     = A\beq;
%     [a2, xi2, rho2, fpval] = extractPrimalVars(x, H, uIsNonZero, Xizero, nAs, zz, D);
%     na    = norm(a - a2);
%     nxi   = norm(xi - xi2);
%     nr    = abs(rho-rho2);
%     if na> 1e-5 || nxi > 1e-5 || nr > 1e-5
%         stophere = 1;
%     end
% end

exitflag = 1;

end




function [a, xi, rho, fpval] = extractPrimalVars(x, H, uIsNonZero, ...
    Xizero, nAs, zz, D, signRho, cheap)

if ~exist('signRho', 'var')
    signRho = 1;
end
if ~exist('cheap', 'var')
    cheap = false;
end

[dim, N] = size(H);
tol = 1e-3;

% Get nonzero a
a = zeros(dim, 1);
a(zz)   = x(1:nAs);

% rho
rho = signRho .* x(end);


%% xi
if ~cheap
    xi = zeros(N,1);
    uIsZero = ~ uIsNonZero;
    xi(uIsZero)    = max(0,rho*ones(sum(uIsZero),1) - H(:,uIsZero)'*a);
    % TEST if ui = 0 => xi = 0
    % if ~all(xi(uIsZero)<tol); 
    %     fprintf('Some zero ui have non zero xi');
    % end

    xiIdx= zeros(N,1);
    xiIdx(uIsNonZero) = Xizero;
    xiIdx = uIsNonZero' & ~xiIdx;
    xi(xiIdx) = x((nAs+1):(end-1));
else
    xi = max(0,rho*ones(N,1) - H'*a);
end

% Primal objective
if sum(a) > 1 + tol || sum(a) < 1 - tol
    fpval = Inf;
else
    fpval = rho - D*sum(xi);
end

end



function out = iterativeUnion(x, initFactor, remFactor, eq_eps)
n = length(x);
[x_s, x_ord] = sort(x);
out = false(n,1);

out(x_ord(1)) = true;
lb = x_s(1)*initFactor;
for i=2:n
    if x_s(i) < lb + eq_eps
        out(x_ord(i)) = true;
        lb = max(lb, x_s(i)*remFactor);
    else
        break
    end
end
end