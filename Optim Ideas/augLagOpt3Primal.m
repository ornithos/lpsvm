function [x, y] = augLagOpt3Primal(f, H, D, x0, maxIter, tol, mu, dblA)
% augLagOpt 3rd Augmented Lagrangian Optimisation of the LPSVM Dual Problem
%   GULER. CONJUGADTE GRAD VERSION (Alg 2.1)
%   This is similar to a Method of Multipliers approach
%   lambda are still stationary

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
%    Warm start?

%% set-up
% ensuring correct orientation of vectors
if size(f,2)>1; f = f'; end;
if any(size(D))>1; error('D must be scalar'); end;
if size(x0,2)>1; x0 = x0'; end;

% admin
[m0, n0] = size(H);
A = [H' eye(n0) -eye(n0) -ones(n0,1) ones(n0,1); ones(1,m0) zeros(1,2*n0+2)];
[m, n] = size(A);

lb = zeros(n,1);
qpOpts = optimoptions(@quadprog, 'Display', 'None', 'TolFun', tol);

% defaults
if isempty(x0); x0 = zeros(n,1); end;
if isempty(mu); mu = 1; end;
if isempty(dblA); dblA = 1; end;

% initialise
currX = zeros(n,1);     % ignoring x0 for now, doesn't seem to help
lambda = ones(m,1);
b = [zeros(m-1,1); 1];
f = [zeros(m0, 1); D*ones(n0,1); zeros(n0,1); -1; +1];
eta = lambda;

%% Main Algorithm
for k = 1:maxIter
    a = 0.5*(sqrt((dblA*mu)^2 + 4*dblA*mu) - dblA*mu);
    z = (1 - a)*lambda + a*eta;
    
%     obj = @(t) f'*t + z'*(b-A*t) + 0.5*lambda.*(A*t - b)'*(A*t - b);
%      [currX, q, exitflag] = quadprog(lambda.*A'*A, f - A'*z - lambda.*A'*b, ...
%          [], [], [], [], lb, [], [], qpOpts);
     
%     currX = SCDNNQuad2a(f,A,b,z,lambda,currX,3000,1e-7);
    currX = SCDNNPrimal2a(f,A,b,z,mu,currX,3000,1e-7);
%     lf = length(f); lb = length(b); Hn = lf - lb  - 1; Hp = 2*lb - lf;
%     currX = SCDNNQuad2b(f,A(1:Hp,1:Hn),A,b,z,lambda, 0.2, currX,4000,1e-7, 1);
    lambda = z + mu*(b-A*currX);
    eta = eta + (lambda - z)./a;
    dblA = (1 - a)*dblA;
    
    % Convergence criteria
    dualityGap = f'*currX-b'*lambda;
    convg = [max(abs(A*currX-b)), max(A'*lambda - f), abs(dualityGap)];
    if all(convg < n*tol)
        break; 
    end
end

if ~all(convg < n*tol)
    fprintf('\n (augLagOpt3) Optimum not reached within %d iters\n', maxIter);
else
    fprintf('\n (augLagOpt3) Converged in %d iters\n', k);
end

x = currX; y = lambda;
end
