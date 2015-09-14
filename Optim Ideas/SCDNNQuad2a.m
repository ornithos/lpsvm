function x = SCDNNQuad2a(f, A, b, y, lambda, x0, maxIter, tol)
% (Stochastic) Coordinate Ascent for non-negative quadratic minimisation.
% (2) We now actually take analytic (optimal) step for each j rather than
% gradient step. Reduced compute time.
%
% Since we expect a sparse vector, we can reduce the probability of
% choosing a coordinate if it doesn't move (much) on an iteration or
% attempts to go negative. This is not well implemented yet.
% Eventually will need to move to C.

% Also look at optimal step size choices

%% set-up
[~, n] = size(A);
if size(b,2)>1; b = b'; end;
if isempty(x0)
    x0 = zeros(n,1);
    x = x0;
elseif size(x0,2)>1
    x = x0';
else
    x = x0;
end
obj = @(t) f'*t + y'*(b-A*t) + 0.5*lambda.*(A*t - b)'*(A*t - b);

% Pre compute useful quantities
A2 = zeros(n,1);
Anb = zeros(n,1);
s = zeros(n,1);
m = zeros(n,1);

for cN = 1:n
    A2(cN) = A(:,cN)'*A(:,cN);       % L2 norm of each column
    Anb(cN) = A(:,cN)'*b;            % A_j' * b
    s(cN) = A(:,cN)'*y - f(cN);      % A_j' * y
    m(cN) = s(cN)./lambda + Anb(cN); % (A_j' * y)/lambda + A_j' * b
end
ATA = A'*A;
ylb = y + lambda.*b;
AxOld = A*x;                         % Not used until calculating obj.

iters = 1:n;
iters = iters(abs(A2)>tol);   % remove any coordinates with zero norm.
% since the coordinate will make no difference to minimum, and DIV0 error.

trig = zeros(2,1);
%% Algorithm: Coordinate Descent
r = ATA*x;
ov = zeros(maxIter,1);
saveX = zeros(n,maxIter);
for cI = 1:maxIter
    
    for cN = iters       

        % Update coord cN
        xOld = x(cN);
        x(cN) = (m(cN) - r(cN))/A2(cN) + x(cN);
        
        % Projection
        if x(cN) < 0
            x(cN) = 0;
        end
        
        % Update ATA*x to reflect new coord of x.
        if abs(x(cN) - xOld) > tol*1e-3
            r = r + ATA(:,cN).*(x(cN) - xOld);
        end
        
    end
    
    % Calculate objective and test for convergence
    AxNew = A*x;
    objVal = calcObjective(x, x0, AxNew, AxOld, ylb, lambda);
    if abs(objVal) < tol
        break
    end
    
    % reset for next iteration
    x0 = x;
    AxOld = AxNew;
    ov(cI) = objVal;
    saveX(:,cI) = x;
    
end

fprintf('(SCDNNQuad2) Iters = %d of %d\n' , cI, maxIter);
end


function obj = calcObjective(x, x0, AxNew, AxOld, ylb, lambda)
    F = x(end-1) - x(end) - x0(end-1) + x0(end) ; % f = [0 0 0 ... 0 1 -1];
    Delta = AxNew - AxOld;
    Linear = ylb'*Delta;
    obj = F - Linear + 0.5*lambda.*(sum(AxNew.^2) - sum(AxOld.^2));
end

% Test with line search
%         d = zeros(n,1); d(cN) = 1;
%         l(cN) = fminbnd(@(l) obj(x+l*d), -1000,1000);
%         l(cN) = x(cN)+l(cN);
%         if l(cN) < 0; l(cN) = 0; end;
%             
%         diff = max(x2 - l);
%         fprintf('Max difference = %2.7f\n',diff);