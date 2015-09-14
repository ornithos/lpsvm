function[out, exitflag, primal, time, msgl] = ...
                    solveLPActiveAB3GS(H, D, options, wrm_strt, NN, ass, dbg)
%SOLVELPACTIVE Test how much difference GS Orthonormalisation can make!

tol = 1e-5;
time = zeros(1, 200);

%Start with random columns
[dim, N] = size(H);
if NN == 0; NN = ceil(N/100); end;
% NN = 10*ceil(0.3/D);

if ceil(1/D) + dim < N
    num_needed = ceil(1/D) + dim;
    
    if ~isempty(wrm_strt)
        % include warm start columns if available
        P = wrm_strt;
        num_needed = num_needed - length(P);
        if num_needed > 0
            remaining = setdiff(1:N,P);
            Hsel = H(:,remaining);
            % SHOULD THINK ABOUT MAKING THESE NEXT CHOICES ORTHOGONAL!
            P = union(P, remaining(chooseInitCol(Hsel, num_needed)));
        end
    else
        % warm start unavailable
        P = chooseInitCol2(H, ceil(1/D) + dim, D);
    end
    Hsel = H(:,P);
else
    % constraints enforce all columns in active set.
    P = 1:size(H,2);
    Hsel = H;
end


fprintf('R2MP: top %d\n', dim);
[x, ~, primal, t1] = solveLPActiveAB1(H, D, options, ...
        P, size(H,1), 0, false); %add top (dim) violations
fprintf('R2MP: wgt orthogonal %d\n', dim);
[x2, ~, primal2, t2] = solveLPActiveAB1(H, D, options, ...
        P, size(H,1), 1, false); %add orthonormal (weighted by violations)
fprintf('R2MP: pure orthogonal %d\n', dim);
[x3, ~, primal3, t3] = solveLPActiveAB1(H, D, options, ...
        P, size(H,1), 2, false); %add orthonormal

n = evalin('base', 'n');
evalin('base',strcat('gsiters(', num2str(n), ',:) = [', num2str(...
    [length(t1) length(t2) length(t3)]), '];'));
assignin('base','n', n+1);

% Set output
out      = x;
time     = [];
exitflag = 1;
msgl     = 0;
end

function I = chooseInitCol(H, NN)

dim = size(H, 1);
cc = -1000000*ones(dim, 1);

msquare = @(x, y) (x-y).^2;
[~, pos] = sort(sum(bsxfun(msquare, cc, H), 1));
I = pos(1:NN);
end



function I = chooseInitCol2(H, NN, D)

rows = size(H,1);
numfill = floor(1/D);
%remainder = 1 - numfill*D;
solosGt0 = zeros(1,size(H,2));
for cI = 1:rows
    [~,ordered] = sort(H(cI,:));
    solosGt0(ordered(1:numfill)) = ...
        solosGt0(ordered(1:numfill)) + ones(1, numfill);
end

% add col total information to break ties
breakties = -sum(H, 1);
quicksearch = min(100, size(H,2));
breakties = breakties ./ (1000 * max(breakties(1:quicksearch)));

[~,ordered] = sort(solosGt0 + breakties, 'descend');

I = ordered(1:NN);
end


function [f, A, b, Aeq, beq, lb, ub] = createProblem(H, D)
[dim, N] = size(H);

f = zeros(1, N+1); f(end) = 1; % Objective function = beta

A   = [H, -ones(dim,1)]; % Hu <= beta
b   = zeros(dim, 1);
Aeq = [ones(1, N) 0]; % sum u = 1
beq = 1;
lb = [zeros(1, N) -Inf]; % u beta
ub = [D*ones(1, N) Inf]; % u beta

end