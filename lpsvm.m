function SVMOut = lpsvm(X, y, D, kernel, initm, maxiter, tol, debug, sinit, ...
                            aZero, calc_lb)
% lpsvm. Main LPSVM function.
% This is an implementation of the algorithm described in the thesis Scaling Kernel
% SVM towards Big Data.
% There's a problem with persistence of the kernel Cache. IF ERROR WITH
% KKMAT FUNCTION, SIMPLY HIT 'clear kernelCache', and this should resolve.
%
%   Arguments:
%   X          -   (Matrix) Design matrix (d x n): each column is an example.
%   y          -   (Vector) Labels of the examples (1 x n). Assumed in {-1,+1}.
%   D          -   (Scalar) upper box constraint
%   kernel     -   (function handle) arbitrary PSD function to calculate kernel.
%   initm      -   (Scalar) Number of learners to initialise procedure with.
%   maxiter    -   (Scalar) Maximum number of epochs (outer iterations).
%   tol        -   (Scalar) Convergence tolerance.
%   debug      -   (Logical) really means verbose.
%   sinit      -   (Scalar) Proportion of examples initialised as 'support vectors'
%                  in initial weak learners.
%   aZero      -   (Logical) remove weak learner i if a_i = 0. **DEPRECATED** It is
%                  uncommon in large problems for any components of a to be zero, and
%                  smaller problems which did result in some a_i = 0, this strategy
%                  performed worse.
%   calc_lb    -   (Logical) Calculate the lower bound exactly (using the SVM dual).
%                  Uses v_j as surrogate if false (see thesis).
%
%
%   Outputs:
%   SVMOut     -   Structure corresponding to SVM model.
%

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
pctActiveSet = 150;

H = zeros(maxiter, N); % Pre-allocate LP matrix
kernTic = tic;
H = initModelMatrix(X, y, model, initm, kernel, H);
time_kern(1) = toc(kernTic);

% Linear Program options
options = optimoptions(@linprog, 'Algorithm', 'dual-simplex', ...
    'Display', 'None');
% options = optimset('Display','none');

% Preallocate for speed up
lowb      = zeros(1, maxiter);
betahist  = zeros(1, maxiter);
testhist  = zeros(1, maxiter);
nSVs      = zeros(1, maxiter);
vjhist    = zeros(1, maxiter+initm);
time_rmp  = zeros(1, maxiter+1);
time_kern = zeros(1, maxiter+2);
time_r2mp = cell(1,maxiter+1);
for i = 1:initm; vjhist(i) = model{i}.v; end
aMem      = NaN(maxiter+initm, maxiter+1);
rowsRemov = 0;
optim = false;

while(iter <= maxiter),
     
     if(iter > 0),
         % Solve new model
         kernTic = tic;
         newModel = solveModel(X, y, u, kernel); % tolerance inside
         time_kern(iter+1) = toc(kernTic);
         
         % Save vj
         vjhist(iter+initm) = newModel.v;
         if debug
            fprintf('\niter %d, duality gap %1.4f, ', iter, newModel.v-beta);
         end
         
         if ~optim && ((newModel.v - tol) <= beta),
             fprintf('\niter %d: at optimum\n', iter);
             maxiter = min(maxiter, iter+20);
             optim = true;
         end

         % early stopping
         if ~optim && (abs(max(-vjhist(1:iter)) - -beta) <= tol),
             fprintf('\niter %d Early stopping\n', iter);
             maxiter = min(maxiter, iter+20);
             optim = true;
         end
         
         % ===== CONVERGENCE TESTING ===============
         if calc_lb
             a = primal.a;
             [nSVs, SVset] = countSV(model, nModels, a);

             % calculate alphas
             wgttol = 1e-5;
             alpha = zeros(1,N);
             for i = 1:nModels
                 if(a(i) > wgttol)
                     idx = model{i}.i;
                     alpha(idx) = alpha(idx) + a(i).*(model{i}.u)./model{i}.v;
                 end
             end

             % separately calculate IsSV
             IsSupportVector = zeros(N,1);
             svnum = 1;
             for i = 1:N
                 if i == SVset(svnum)
                     IsSupportVector(i) = 1;
                     svnum = svnum + 1;
                 end
                 if(svnum > nSVs); break; end
             end

             % Master Problem Dual
             tmpModel = solveModel(X, y, alpha, kernel);
             lowb(iter) = tmpModel.v^2;
         else
             lowb(iter) = 0;
         end
         % ========================================
         % Add model
         model{iter+initm}       = newModel;
         nModels                 = iter+initm;
         H(nModels-rowsRemov, :) = newModelRow(X, y, model, iter+initm, kernel);
     end
     
     if ~isreal(H)
         error('Kernel Cache screwed up. Please reset kernel cache;');
     end
     % Solve LP
     nModels = iter+initm;
     lastActive = model{nModels}.i(model{nModels}.u>1e-5);
     
     outerTic = tic;
     [x, exitflag, primal, t] = solveLPActiveAB3(H(1:nModels, :), ...
         D, options, lastActive, pctActiveSet,0, false, debug); % ass = (-1, all), (0, top), (1, idpdt)
%      [x2, exitflag, primal2, t, r2_ml] = solveLPActiveAB2(H(1:nModels, :), ...
%          D, options, lastActive, pctActiveSet,0, false, debug); 
%      stophere = 0;
%      if max(abs(x-x2))>1e-7
%          stophere = 1;
%      elseif  max(abs(primal.a-primal2.a))>1e-7
%          stophere = 2;
%      elseif max(abs(primal.rho-primal2.rho))>1e-7
%          stophere = 3;
%      elseif max(abs(primal.xi-primal2.xi))>1e-7
%          stophere = 4;
%      end
%      if stophere > 0
%          stophere = 1;
%      end
     
     time_r2mp{iter+1} = t;
     time_rmp(iter+1) = toc(outerTic);
   
     if(exitflag ~= 1),
         exitflag
     end
     
     %Calculate value in primal
     if debug; fprintf('%2.4f', D*sum(primal.xi) - primal.rho); end;
     aMem(1:(iter+initm), iter+1) = primal.a;
     
     % new vars
     beta = x(end);
     u    = x(1:(end-1))';
     
     % CHOOSING TOP N - TEMPORARY
%      if ismember(iter,[3,10,20])
%          activeSetTesting(H(1:nModels, :), D, options, lastActive, debug, iter, u);
%          if iter == 20
%              SVMOut=[];
%             return
%          end
%      end
     
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
% [a, ~] = solveWeights(beta, u, H(1:nModels, :), D, 'nnls');
% [f, A, b, Aeq, beq, lb, ub] = createPrimalProblem(Hsel, D);
% [xP, ~ , exitflag, ~, ~] = linprog(f,A,b,Aeq,beq,lb, ub, [], options);
% if exitflag ~=1
%     fprintf('Final optimisation failed! Code paused.\n');
%     Pause
% end
% a = xP(1:dim);
a = primal.a;

[nSVs, SVset] = countSV(model, nModels, a);

% Prune if necessary
betahist = [NaN(1, initm) betahist(1:iter)];
lowb = [NaN(1, initm) lowb(1:iter)];
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
SVMOut.time_kern = time_kern;
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
SVMOut.lowb = lowb;
end

