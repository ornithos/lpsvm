function x = SCDNNQuad3(f, A, b, y, lambda, x0, maxIter, tol)
% (Stochastic) Coordinate Ascent for non-negative quadratic minimisation.
% (3) Optimal step plus adaptive stochastic choice of coordinate
%
% Since we expect a sparse vector, we can reduce the probability of
% choosing a coordinate if it doesn't move (much) on an iteration or
% attempts to go negative. This is not well implemented yet.
% Eventually will need to move to C.

%% set-up
[m, n] = size(A);
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
objVal = 1;                          % Used for reweighting criterion
% softmax probability weights
weights = single(abs(A2)>tol);

saveX = zeros(n,maxIter); saveW = zeros(n, maxIter);
%% Algorithm: Coordinate Descent
r = ATA*x;
for cI = 1:maxIter
    
    weightsNext = weights;
    for cN = 1:(0.5*n)       % check convergence every n coordinate updates     

        % choose coordinate
        rN = logical(mnrnd(1, weightSoftmax(weights)));
        
        % Update coord rN
        xOld = x(rN);
        x(rN) = (m(rN) - r(rN))/A2(rN) + x(rN);
        
        % Projection
        if x(rN) < 0
            x(rN) = 0;
        end
        
        % Update weights
        coodMove = abs(x(rN) - xOld);
        if coodMove > 0.01
            weightsNext(rN) = max(0.125, min(4, weights(rN)*2));
        elseif weights(rN) > 1
            weightsNext(rN) = max(0.125, min(4, weights(rN)/2));
        elseif coodMove < tol
            weightsNext(rN) = max(0.125, min(4, weights(rN)/2));
        end
        weights(rN) = -Inf; 

        % Update ATA*x to reflect new coord of x.
        if coodMove > tol*1e-3
            r = r + ATA(:,rN).*(x(rN) - xOld);
        end
    end
    
    % Calculate objective and test for convergence
    AxNew = A*x;
    objVal = calcObjective(x, x0, AxNew, AxOld, ylb, lambda);
    if abs(objVal) < tol
        break
    end
    
    saveX(:,cI) = x; saveW(:,cI) = weightsNext;
    % reset for next iteration
    x0 = x;
    AxOld = AxNew;
    weights = weightsNext;
end

fprintf('(SCDNNQuad3) Iters = %d of %d\n' , cI, maxIter);
end


function obj = calcObjective(x, x0, AxNew, AxOld, ylb, lambda)
    F = x(end-1) - x(end) - x0(end-1) + x0(end) ; % f = [0 0 0 ... 0 1 -1];
    Delta = AxNew - AxOld;
    Linear = ylb'*Delta;
    obj = F - Linear + 0.5*lambda.*(sum(AxNew.^2) - sum(AxOld.^2));
end


% x should be an n x 2 matrix, second column being the (int) power of 2
function x = weightUpdate(x, increase)
    % increase should be either true or false
    ceil = 4;
    floor = 0.125;
    if increase
        if pow2 < ceil
            x = x*2;
        end
    else
        if pow2 > floor
            x = x/2;
        end
    end
end

function x = weightSoftmax(x)
    x = exp(x);
    x = x./sum(x);
end





function x = softmax2(num ,denom)
    x = exp(num)./denom;
end

function denom = updateSoftmax2(denom, rm, add)
    denom = denom - exp(rm) + exp(add);
end