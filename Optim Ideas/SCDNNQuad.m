function x = SCDNNQuad(f, A, b, y, lambda, x0, maxIter, tol, gam0, pow)
% Stochastic Coordinate Ascent for non-negative quadratic minimisation.
% Since we expect a sparse vector, we can reduce the probability of
% choosing a coordinate if it doesn't move (much) on an iteration or
% attempts to go negative. This is not well implemented yet.
% Eventually will need to move to C.

% Also look at optimal step size choices

%% set-up
[m, n] = size(A);
if size(b,2)>1; b = b'; end;
if isempty(pow); pow = -1; end;
if isempty(x0)
    x0 = zeros(n,1);
    x = x0;
elseif size(x0,2)>1
    x = x0';
else
    x = x0;
end
obj = @(t) f'*t + y'*(b-A*t) + 0.5*lambda.*(A*t - b)'*(A*t - b);

%% Algorithm: Coordinate Descent
for cI = 1:maxIter
    step = gam0/(cI.^pow);
    gcd = zeros(1,n);
    for cN = 1:n
        gcd(cN) = f(cN) - A(:,cN)'*y + lambda.*A(:,cN)'*(A*x - b);
        x(cN) = x(cN) - step.*gcd(cN);
        
        % Projection
        if x(cN) < 0
            x(cN) = 0;
        end
    end
    
    % Convergence (This is a bit poor..!)
    %fprintf('%2.5f, shortcut: %2.5f\n', obj(x), f'*(x-x0));
    if abs(f'*(x-x0)) < tol
        break
    end
    x0 = x;
end

end


    % batch
    %g = (f - A'*y - lambda.*A'*b + lambda.*A'*A*x);
    
    % finite differences
%     obj = @(t) f'*t + y'*(b-A*t) + 0.5*lambda.*(A*t - b)'*(A*t - b);
%     delta = 1e-9;
%     g2 = zeros(n,1);
%     for cN = 1:n
%         d = zeros(n,1);
%         d(cN) = delta;
%         g2(cN) = (obj(x) - obj(x-d))/delta;
%     end
%     fprintf('Max Discrepancy: %1.8f\n', max(abs(g - g2)));

