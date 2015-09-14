function x = SCDNNQuad2(f, A, b, y, lambda, x0, maxIter, tol)
% (Stochastic) Coordinate Ascent for non-negative quadratic minimisation.
% (2) We now actually take analytic (optimal) step for each j rather than
% gradient step.
%
% Since we expect a sparse vector, we can reduce the probability of
% choosing a coordinate if it doesn't move (much) on an iteration or
% attempts to go negative. This is not well implemented yet.
% Eventually will need to move to C.

% Also look at optimal step size choices

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
notJ = logical(ones(n,n) - eye(n));
AnRest = zeros(n,n-1);
Anb = zeros(n,1);
s = zeros(n,1);
m = zeros(n,1);

for cN = 1:n
    A2(cN) = A(:,cN)'*A(:,cN);
    AnRest(cN,:) = A(:,cN)'*A(:,notJ(:,cN));
    Anb(cN) = A(:,cN)'*b;
    s(cN) = A(:,cN)'*y - f(cN);
    m(cN) = s(cN)./lambda + Anb(cN);
end
iters = 1:n;
iters = iters(abs(A2(cN))>tol);   % remove any coordinates with zero norm.
% since the coordinate will make no difference to minimum, and DIV0 error.


%% Algorithm: Coordinate Descent
for cI = 1:maxIter
    for cN = 1:n       
        if(abs(A2(cN))>tol)
            AnjXnj = AnRest(cN,:)*x(notJ(:,cN));
            x(cN) = (m(cN) - AnjXnj)./A2(cN); 
        end

        % Projection
        if x(cN) < 0
            x(cN) = 0;
        end
    end
    
    % Convergence (This is a bit poor..!)
    %fprintf('%2.5f, shortcut: %2.5f\n', obj(x), abs(f'*(x-x0)));
    if abs(f'*(x-x0)) < tol
        break
    end
    x0 = x;
end

fprintf('(SCDNNQuad2) Iters = %d of %d\n' , cI, maxIter);
end

% 'Clever' efficient updates - but do not take into account new coords
% l = zeros(n,1);
% x2 = zeros(n,1);
%         r = ATA*x;
%         ATA = A'*A;
%         x2(cN) = (m(cN) - r(cN))/A2(cN) + x(cN);


% Test with line search
%         d = zeros(n,1); d(cN) = 1;
%         l(cN) = fminbnd(@(l) obj(x+l*d), -1000,1000);
%         l(cN) = x(cN)+l(cN);
%         if l(cN) < 0; l(cN) = 0; end;
%             
%         diff = max(x2 - l);
%         fprintf('Max difference = %2.7f\n',diff);