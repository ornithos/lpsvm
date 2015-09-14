function SVMOut = lpsvmAD(X, y, D, kernel, initm, maxiter, npar, npiter, ...
                tol, debug, sinit, aZero)
%LPSVM SVM implementation based on LPSolver via LPBoosting
% uses ADMM Solver --> Slower but parallelisable

iter = 0;
[~, N] = size(X);
r_ml = 0;   % RMP msg length: for writing over messages in terminal
r2_ml = 0;  % R2MP msg length:   ----- " ------ " ------

% Setting up of the optimization problem
model    = cell(1, maxiter+initm);
model    = initModels(X, y, initm, kernel, sinit, model);
beta     = 0;

% min amt added in each inner iteration: 30
if N < 10000
    pctActiveSet = 30;
else
    pctActiveSet = 0.003;
end

% Kernel arrays
cleared = false;
while(~cleared); cleared = kernelCache(-1); end % Remove all cached kernel rows
kernelRowsAccessed(0, size(X,2));  % Reset access record array

H = zeros(maxiter, N); % Pre-allocate LP matrix
H = initModelMatrix(X, y, model, initm, kernel, H);

% Linear Program options
options = optimoptions(@linprog, 'Algorithm', 'dual-simplex', ...
    'Display', 'None');
% options = optimset('Display','none');

% Preallocate for speed up
betahist  = zeros(1, maxiter);
testhist  = zeros(1, maxiter);
nSVs      = zeros(1, maxiter);
vjhist    = zeros(1, maxiter+initm);
time_rmp  = zeros(1, maxiter+1);
time_r2mp = cell(1,maxiter+1);
for i = 1:initm; vjhist(i) = model{i}.v; end
aMem      = NaN(maxiter+initm, maxiter+1);
rowsRemov = 0;
     
while(iter <= maxiter),
     
     if(iter > 0),
         % Solve new model
         newModel = solveModel(X, y, u, kernel); % tolerance inside
         
         % Save vj
         vjhist(iter+initm) = newModel.v;
         if debug
            r_ml= fprintf('\niter %d, duality gap %1.4f, ', iter, newModel.v-beta);
         end
             
         if ((newModel.v - tol) <= beta),
             fprintf('\nFull problem achieved: at optimum\n');
             break;
         end

         % early stopping
         if (abs(max(-vjhist(1:iter)) - -beta) <= tol),
             fprintf('\nEarly stopping\n');
             break;
         end
         
         % Add model
         model{iter+initm}       = newModel;
         nModels                 = iter+initm;
         H(nModels-rowsRemov, :) = newModelRow(X, y, model, iter+initm, kernel);
     end
     
     if ~isreal(H)
         error('Kernel Cache screwed up. Please hit kernelCache(-1);');
     end
     % Solve LP
     nModels = iter+initm;
     lastActive = model{nModels}.i(model{nModels}.u>1e-5);
     
     outerTic = tic;
     [x, exitflag, primal, t, r2_ml] = solveLPActiveAD1(H(1:nModels, :), ...
           D, options, lastActive, pctActiveSet,0, npar, npiter, 1e-5, ...
           false, debug);
     time_r2mp{iter+1} = t;
     time_rmp(iter+1) = toc(outerTic);
   
%      if(exitflag ~= 1),
%          exitflag
%      end
     
     %Calculate value in primal
     if debug; fprintf('%2.4f', D*sum(primal.xi) - primal.rho); end;
     aMem(1:(iter+initm), iter+1) = primal.a;
     
     % new vars
     beta = x(end);
     u    = x(1:(end-1))';
     
     % Remove rows for which a == 0?
     aIsZero = primal.a < 1e-7;
     if sum(aIsZero) > 0
        if aZero == 1
            H(aIsZero,:) = [];
            rowsRemov = rowsRemov + sum(aIsZero);
        end
     end
        
     % DEBUG: CHECK R2MP = RMP
%      [dim, N] = size(H(1:nModels, :));
%      [f, A, b, Aeq, beq, lb, ub] = createPrimalProblem(H(1:nModels, :), D);
%      [orSol, fpval , exitflag, ~, lambda] = linprog(f,A,b,Aeq,beq, ...
%          lb, ub, [], optimset('Display','none'));
%      orA = orSol(1:dim);
%      orXi = orSol((dim+1):(end-1));
%      orRho = orSol(end);
%      if abs(-beta-fpval) > 1e-4; fprintf('RCMP obj: %1.4f, %1.4', fpval, -beta); end;
%      if abs(primal.rho - orRho) > 1e-4 fprintf('RCMP rho: %1.4f, %1.4', orRho, primal.rho); end;
%      if max(abs(primal.a - orA)) > 1e-4 fprintf('RCMP a max err: %1.4f', max(abs(primal.a - orA))); end;
%      if max(abs(primal.xi - orXi(primal.i))) > 1e-4 fprintf('RCMP xi max err: %1.4f', max(abs(primal.xi - orXi(primal.i)))); end;
     % =================================================
     
     % Increase iteration
     iter = iter+1;  
     
     % History of duality gap
     betahist(iter) = beta;
end

% Exhausting loop rather than breaking requires these for consistency:
if(iter > maxiter)
    iter = iter-1;
    if debug; fprintf('\n'); end;
end

% Solve weighting
a = primal.a;
[nSVs, SVset] = countSV(model, nModels, a);

% Prune if necessary
betahist = [NaN(1, initm) betahist(1:iter)];
vjhist   = vjhist(1:(iter+initm));
model    = model(1:nModels);
time_rmp     = time_rmp(1:iter);


% Calculate actual weights of SVs for prediction
wgttol = 1e-5;
alpha = zeros(1,N);
for i = 1:nModels
    if(a(i) > wgttol)
        idx = model{i}.i;
        alpha(idx) = alpha(idx) + a(i).*(model{i}.u).*y(idx)./model{i}.v;
    end
end

% Create SVM model object
SVMOut.model = model;
SVMOut.time = time_rmp;
SVMOut.time_inner = time_r2mp;
SVMOut.a = a;
SVMOut.iter = iter;
SVMOut.betahist = betahist;
SVMOut.vjhist = vjhist;
SVMOut.vjbest = newModel.v;
SVMOut.svset = SVset;

SVMOut.IsSupportVector = zeros(N,1);
svnum = 1;
for i = 1:N
    if i == SVset(svnum)
        SVMOut.IsSupportVector(i) = 1;
        svnum = svnum + 1;
    end
    if(svnum > nSVs); break; end
end
SVMOut.IsSupportVector = logical(SVMOut.IsSupportVector);
SVMOut.Alpha = alpha(SVMOut.IsSupportVector);
SVMOut.SVs = X(:, SVMOut.IsSupportVector);
SVMOut.kernel = kernel;
SVMOut.aMem = aMem;

end

