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
[~, n] = size(A);
A = [A; -eye(n)];
Am = size(A,1);

if size(b,2)>1; b = b'; end;
if isempty(x0)
    x0 = zeros(Am,1);
    x = x0;
elseif size(x0,2)>1
    x = x0';
else
    x = x0;
end
obj = @(t) b'*x - lambda'*(A'*t + f) + 0.5*mu.*norm(A'*t + f)^2;

% Pre compute useful quantities
A2 = zeros(Am,1);
for cM = 1:Am;  A2(cM) = A(cM,:)*A(cM,:)'; end;
Almuf = A*(lambda - mu.*f);
m = (Almuf - b) ./ (mu.*A2); 

AAT = A*A';


%% Algorithm: Coordinate Descent
r = AAT*x;
AxOld = A'*x;
ov = zeros(maxIter,1);
saveX = zeros(cM,maxIter);
for cI = 1:maxIter
    
    for cM = 1:Am      

        % Update coord cN
        xOld = x;
        x(cM) = m(cM) - r(cM)/A2(cM) + x(cM);
        
        % Test with line search
        d = zeros(Am,1); d(cM) = 1;
        l = fminbnd(@(l) obj(xOld+l*d), -1000,1000);
        diff = max(x(cM) - (xOld(cM) + l));
        if diff > 1e-9
            fprintf('Max difference = %2.7f\n',diff);
        end

        % Projection
        if x(cM) < 0
            x(cM) = 0;
        end
        
        % Update ATA*x to reflect new coord of x.
        if abs(x(cM) - xOld) > tol*1e-3
            r = r + AAT(:,cM).*(x(cM) - xOld(cM));
        end
        
    end
    
    % Calculate objective and test for convergence
    AxNew = A'*x;
    objVal = calcObjective(x, x0, b, AxNew, AxOld, lambda, mu, f);
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


function obj = calcObjective(x, x0, b, AxNew, AxOld, lambda, mu, f)
    F = b'*x - b'*x0;
    Delta = AxNew - AxOld;
    Linear = (lambda - mu.*f)'*Delta;
    obj = F - Linear + 0.5*mu.*(sum(AxNew.^2) - sum(AxOld.^2));
end

