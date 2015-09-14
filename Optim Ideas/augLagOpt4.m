function [x, y] = augLagOpt4(f, A, b, x0, maxIter, tol, mu, kappaOpt)
% augLagOpt 3rd Augmented Lagrangian Optimisation of the LPSVM Dual Problem
%   NOCEDAL & WRIGHT (2006) Algorithm 17.4 Box Constrained Lagrangian
%   Method. Also the same framework used by LANCELOT package.

%   Arguments:
%   f          -   (Col Vector) Objective function
%   A          -   (Matrix) equality constraint for Ax = b
%   b          -   (Col Vector) equality constraint for Ax = b
%   x0         -   Initial (warm start) x
%   maxIter    -   maximum number of iterations in FW loop
%   tol        -   stopping criterion.
%   eps        -   tolerance for determining equality

% IGNORING SPARSE LINEAR ALGEBRA FOR NOW
%    SPARSE ALGEBRA

%% set-up
% ensuring correct orientation of vectors
if size(f,2)>1; f = f'; end;
if size(b,2)>1; b = b'; end;
if size(x0,2)>1; x0 = x0'; end;

% admin
[m, n] = size(A);
lb = zeros(n,1);

% defaults
if isempty(x0); x0 = zeros(n,1); end;
if isempty(mu); mu = 1; end;
if isempty(kappaOpt); kappaOpt = 1e-8; end;    % currently unused - using diff cvg. criterion.
omega = 1/mu;
kappa = 0.1;

% initialise
currX = x0;
lambda = b;

%% Main Algorithm
for k = 1:maxIter
    
    % --- Optimisation ----------------------------
    qpOpts = optimoptions(@quadprog, 'Display', 'None', 'TolFun', omega);
%     if ~issymmetric(mu.*A'*A); 
%         fprintf('bad hessian detected\n'); 
%     end;
%     [currX, fOpt, exitflag] = quadprog(mu.*(A'*A), f - A'*lambda - mu.*A'*b, ...
%          [], [], [], [], lb, [], [], qpOpts);
     
%     currX = SCDNNQuad2a(f,A,b,lambda,mu,currX,3000,omega);
    lenf = length(f); lenb = length(b); Hn = lenf - lenb  - 1; Hp = 2*lenb - lenf;
    currX = SCDNNQuad2b(f,A(1:Hp,1:Hn),A,b,lambda,mu, 0.2, currX,4000,omega,1);
    % ---------------------------------------------
    
    % update branching
    constraints = A*currX - b;
    if norm(constraints) < kappa
        
        % Convergence criteria
        dualityGap = f'*currX-b'*lambda;
        convg = [max(abs(A*currX-b)), max(A'*lambda - f), abs(dualityGap)];
        if all(convg < n*tol)
            break; 
        end
        
        % else update lagrange multipliers, tighten tolerances
        lambda = lambda - mu*constraints;
      % mu = mu;   % not updated in this branch
        kappa = kappa/(mu.^0.9);
        omega = omega/mu;
    else
        % increase penalty, tighten tolerances
      % lambda = lambda;  % not updated in this branch
        mu = mu*10;      % (N&W: *100 - seen some singularity issues here)
        kappa = 1/(mu.^0.1);
        omega = 1/mu;
    end
end

if ~all(convg < n*tol)
    fprintf('\n (augLagOpt4) Optimum not reached within %d iters\n', maxIter);
else
    fprintf('\n (augLagOpt4) Converged in %d iters\n', k);
end

x = currX; y = lambda;
end