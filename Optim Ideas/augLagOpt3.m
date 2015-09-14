function [x, y] = augLagOpt3(f, A, b, x0, maxIter, tol, lambda, dblA)
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
if size(b,2)>1; b = b'; end;
if size(x0,2)>1; x0 = x0'; end;

% admin
[m, n] = size(A);
lb = zeros(n,1);
qpOpts = optimoptions(@quadprog, 'Display', 'None', 'TolFun', tol);

% defaults
if isempty(x0); x0 = zeros(n,1); end;
if isempty(lambda); lambda = 1; end;
if isempty(dblA); dblA = 1; end;

% initialise
currX = x0;
currY = zeros(m,1);
currY = yUpdate(currY, currX, A, b, lambda);
eta = currY;

%% Main Algorithm
for k = 1:maxIter
    a = 0.5*(sqrt((dblA*lambda)^2 + 4*dblA*lambda) - dblA*lambda);
    z = (1 - a)*currY + a*eta;
    
    obj = @(t) f'*t + z'*(b-A*t) + 0.5*lambda.*(A*t - b)'*(A*t - b);
     [currX, q, exitflag] = quadprog(lambda.*A'*A, f - A'*z - lambda.*A'*b, ...
         [], [], [], [], lb, [], [], qpOpts);
     
%       currX = SCDNNQuad2a(f,A,b,z,lambda,currX,3000,1e-7);
%     currX = SCDNNQuad2a(f,A,b,z,lambda,currX,3000,1e-7);
%     lf = length(f); lb = length(b); Hn = lf - lb  - 1; Hp = 2*lb - lf;
%     currX = SCDNNQuad2b(f,A(1:Hp,1:Hn),A,b,z,lambda, 0.2, currX,4000,1e-7, 1);
    currY = yUpdate(z, currX, A, b, 1);
    eta = eta + (currY - z)./a;
    dblA = (1 - a)*dblA;
    
    % Convergence criteria
    dualityGap = f'*currX-b'*currY;
    convg = [max(abs(A*currX-b)), max(A'*currY - f), abs(dualityGap)];
    if all(convg < n*tol)
        break; 
    end
end

if ~all(convg < n*tol)
    fprintf('\n (augLagOpt3) Optimum not reached within %d iters\n', maxIter);
else
    fprintf('\n (augLagOpt3) Converged in %d iters\n', k);
end

x = currX; y = currY;
end

function y = yUpdate(yOld, x, A, b, lambda)
    y = yOld + lambda*(b-A*x);
end