function x = SCDNNPrimal2a(f, A, b, lambda, mu, x0, maxIter, tol)
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
[Am, n] = size(A);

if size(b,2)>1; b = b'; end;
if isempty(x0)
    x0 = zeros(Am,1);
    x = x0;
elseif size(x0,2)>1
    x = x0';
else
    x = x0;
end
obj = @(t) f'*t + lambda'*(b-A*t) + 0.5*mu.*(A*t - b)'*(A*t - b);

% Pre compute useful quantities
A2 = zeros(n,1);
for cN = 1:n;  A2(cN) = A(:,cN)'*A(:,cN); end;
lpb = lambda + mu.*b;
m = (A'*lpb - f) ./ (mu.*A2); 

ATA = A'*A;


%% Algorithm: Coordinate Descent
r = ATA*x;
AxOld = A*x;
ov = zeros(maxIter,1);
saveX = zeros(n,maxIter);
for cI = 1:maxIter
    
    for cN = 1:n      

        % Update coord cN
        xOld = x;
        x(cN) = m(cN) - r(cN)/A2(cN) + x(cN);
        
        % Test with line search
        d = zeros(n,1); d(cN) = 1;
        l = fminbnd(@(l) obj(xOld+l*d), -1000,1000);
        diff = max(abs(x(cN) - (xOld(cN) + l)));
        if diff > 1e-9
            fprintf('Max difference = %2.7f\n',diff);
        end

        % Projection
        if x(cN) < 0
            x(cN) = 0;
        end
        
        % Update ATA*x to reflect new coord of x.
        if abs(x(cN) - xOld(cN)) > tol*1e-3
            r = r + ATA(:,cN).*(x(cN) - xOld(cN));
        end
        
    end
    
    % Calculate objective and test for convergence
    AxNew = A*x;
    objVal = calcObjective(x, x0, f, AxNew, AxOld, lpb, mu);
    if abs(objVal) < tol
        break
    end
    
    % reset for next iteration
    x0 = x;
    AxOld = AxNew;
    ov(cI) = objVal;
    saveX(:,cI) = x;
    
end

fprintf('(SCDNNPrimal2) Iters = %d of %d\n' , cI, maxIter);
end


function obj = calcObjective(x, x0, f, AxNew, AxOld, lmb, mu)
    F = f'*(x - x0);
    Delta = AxNew - AxOld;
    Linear = lmb'*Delta;
    obj = F - Linear + 0.5*mu.*(sum(AxNew.^2) - sum(AxOld.^2));
end

