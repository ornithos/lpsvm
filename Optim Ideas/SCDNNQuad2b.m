function x = SCDNNQuad2b(f, H, A, b, y, lambda, D, x0, maxIter, tol, criterion)
% (Stochastic) Coordinate Ascent for non-negative quadratic minimisation.
% (2) We now actually take analytic (optimal) step for each j rather than
% gradient step. Reduced compute time.
%
% Since we expect a sparse vector, we can reduce the probability of
% choosing a coordinate if it doesn't move (much) on an iteration or
% attempts to go negative. This is not well implemented yet.
% Eventually will need to move to C.

% Also look at optimal step size choices

%% set-up for basic updates
% admin ----------------------------
[~, n] = size(A);
if size(b,2)>1; b = b'; end;

if isempty(criterion)   % stopping criterion: norm = 1, objective = 0
    criterion = true;
elseif isnumeric(criterion) || islogical(criterion);
    criterion = logical(criterion);
elseif strcmpi(criterion(1:3), 'obj')
    criterion = 0;
else
    criterion = 1;
end

if isempty(x0)
    x0 = zeros(n,1);
    x = x0;
elseif size(x0,2)>1
    x = x0';
else
    x = x0;
end
xPrev = Inf(size(x0));
obj = @(t) f'*t + y'*(b-A*t) + 0.5*lambda.*(A*t - b)'*(A*t - b);
% ---------------------------------

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

%% set-up for optimised updates

[p, Hn] = size(H);
HTH = H'*H;
H2 = diag(HTH);
constU1 = zeros(1,Hn);
for cN = 1:Hn          % note below that b(end) should be 1 unless I'm playing
    constU1(cN) = (D + b(end) + (H(:,cN)'*y(1:p) + y(p+cN) + y(end))/lambda)/(H2(cN)+2);
end

%% Algorithm: Coordinate Descent
r = ATA*x;
ov = zeros(maxIter,1);
saveX = zeros(n,maxIter);
for cI = 1:maxIter
    if cI > 500
        stophere = 0;
    end
    
    for cN = iters       
        
        % Update coord cN
        xOld = x(cN);
        x(cN) = minimiseCoordinate(x, cN, H, HTH, Hn, p, constU1, D, y, lambda);

        % Projection
        if x(cN) < 0
            x(cN) = 0;
        end
        
    end
    
    % Calculate objective and test for convergence
    if criterion
        objVal = norm(xPrev - x);
        if norm(xPrev - x) <= tol*sqrt(n)*0.01
            break
        end
    else
        AxNew = A*x;
        objVal = calcObjective(x, x0, AxNew, AxOld, ylb, lambda);
        if abs(objVal) < tol
            break
        end
        AxOld = AxNew;
    end
    
    % reset for next iteration
    x0 = x;
    ov(cI) = objVal;
    saveX(:,cI) = x;
    xPrev = x;
    
end

fprintf('(SCDNNQuad2) Iters = %d of %d\n' , cI, maxIter);
if cI > 1200
    stophere = 1;
    %semilogy(1:cI, ov(1:cI));
end
end


function obj = calcObjective(x, x0, AxNew, AxOld, ylb, lambda)
    F = x(end-1) - x(end) - x0(end-1) + x0(end) ; % f = [0 0 0 ... 0 1 -1];
    Delta = AxNew - AxOld;
    Linear = ylb'*Delta;
    obj = F - Linear + 0.5*lambda.*(sum(AxNew.^2) - sum(AxOld.^2));
end

function out = minimiseCoordinate(x, idx, H, HTH, Hn, p, constU1, D, y, lambda)
    if idx <= Hn
        out = constU1(idx) - ((HTH(:,idx)+ones(Hn,1))'*x(1:Hn) + ...
            H(:,idx)'*x((Hn+1):(Hn+p)) - sum(H(:,idx))*(x(end-1)-x(end)) + ...
            x(idx) + x(idx+Hn+p))/(HTH(idx, idx)+2) + x(idx);
    elseif idx <= Hn+p
        out = y(idx-Hn)/lambda - H(idx-Hn,:)*x(1:Hn) + x(end-1)-x(end);
    elseif idx <= 2*Hn+p
        out = D + y(idx-Hn)/lambda - x(idx-Hn-p);
    elseif idx == 2*Hn + p + 1
        out = -mean(y(1:p))/lambda + mean(H*x(1:Hn)) +...
             mean(x((Hn+1):(Hn+p))) + x(end) - 1/(lambda*p);
    elseif idx == 2*Hn + p + 2
        out = mean(y(1:p))/lambda - mean(H*x(1:Hn)) - ...
           mean(x((Hn+1):(Hn+p))) + x(end-1) + 1/(lambda*p);
    else
        error('(SCDNNQuad2b) Invalid Coordinate %d, max %d', idx, 2*Hn + p + 2);
    end
end


% Test with line search
%         d = zeros(n,1); d(cN) = 1;
%         l(cN) = fminbnd(@(l) obj(x+l*d), -1000,1000);
%         l(cN) = x(cN)+l(cN);
%         if l(cN) < 0; l(cN) = 0; end;
%             
%         diff = max(x2 - l);
%         fprintf('Max difference = %2.7f\n',diff);