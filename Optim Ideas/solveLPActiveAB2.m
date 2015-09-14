function[out, exitflag, primal, time, msgl] = ...
                    solveLPActiveAB2(H, D, bPrimal, tradeoff)
% SOLVELPACTIVEAB2 Solve LP approximately with greedy heuristic. Solving
% the LP exactly may not be necessary, since the new rows H_j created are
% only assumed to be weak learners, NOT the optimal rows to add.

% Select columns with greedy heuristic
[~, N] = size(H);
P = chooseInitCol2(H, ceil(1/D), D, tradeoff);
beta = max(H(:,P)*ones(length(P),1).*D);

% Set output
if bPrimal
    [a, rho, xi, fpval] = solveWeights(beta, u, Hsel, D, 'nnls');
    primal.a = a;
    primal.rho = rho;
    primal.xi = zeros(1,N);
    primal.xi(P) = xi;
    primal.fpval = fpval;
    primal.i = P;
else
    primal = [];
end

out      = zeros(N+1, 1);
out(P)   = D;
out(end) = beta;
time = [];
exitflag = 1;

msgl = fprintf('R2MP iter = heur, cols_sel = %d', length(P));
if bPrimal; fprintf(', Primal: '); end
end


function I = chooseInitCol2(H, NN, D, tradeoff)

%% Choose column strategy: use columns in descending order based on
% how many of the solo problems they are nonzero in. In the final subset
% (ie when nonzero == z), we have more tied candidates than spaces. At this
% stage we choose greedily by maximising the minimum in a sequential way.

% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% IMPORTANT: ASSUMES D DIVIDES 1 EXACTLY - TRUE IF USING NU FORMULATION
% it is not too difficult to modify the function if this is false

[rows, cols] = size(H);
numfill = floor(1/D);
%remainder = 1 - numfill*D;
numSoloSolns = zeros(1,cols);
for cJ = 1:rows
    [~,ordered] = sort(H(cJ,:));
    numSoloSolns(ordered(1:numfill)) = ...
        numSoloSolns(ordered(1:numfill)) + ones(1, numfill);
end

% Add all cols with the highest hit rate into solution. But if we cannot
% use all the solutions of a level (since maxAvail > NN), we choose
% greedily to maximise the current minimum.

[~,ordered] = sort(numSoloSolns, 'descend');
tbl = tabulate(numSoloSolns);

maxAvail = cumsum(flipud(tbl(:,2)));
[~,greedyPos] = max(maxAvail >= NN);    % Index of max returns *first* match
greedyEnd = maxAvail(greedyPos);

if greedyPos > 1
    greedyStart = maxAvail(greedyPos-1)+1;
    I = ordered(1:(greedyStart-1));
    runningTotals = H(:,I)*ones(length(I),1).*D;
else
    greedyStart = 1;
    I = [];
    runningTotals = zeros(rows,1);
end

greedyCnt = NN-greedyStart+1;
I2 = zeros(greedyCnt, 1, 'uint32');

%% GREEDY
% Set up orders for each j
orderGreedy = cell(rows,1);
orderCols = cell(rows,1);
ptrGreedy = ones(rows,1);

% define bi-objective ordering on both gradient for {ij} and difference
% across the row {.j} (Balance).
Hsort = H(:,ordered(greedyStart:greedyEnd));
Balance = zeros(size(Hsort));
for cJ = 1:rows
    notJ = setdiff(1:rows, cJ);
    Balance(cJ,:) = sum(Hsort(notJ,:),1)./(rows-1) - Hsort(cJ,:);
end

% orderGreedy{j} is the order in which to solve problem j if problem j was
% the only problem. orderCols{j} is the position in which you find an index
% i in each orderGreedy{j}.
[~, Hsort] = sort(Hsort + tradeoff.*Balance, 2);  
for cJ = 1:rows
    orderGreedy{cJ} = Hsort(cJ,:);
    [~,orderCols{cJ}] = sort(orderGreedy{cJ}, 'ascend');
end

% In each iteration, find the maximum j of {H*u}_j, and seek to minimise by 
% choosing the next column to add in as the one which will shrink this
% quantity the most.
for cI = 1:greedyCnt
    [~,cMax] = max(runningTotals);
    nextCol = orderGreedy{cMax}(ptrGreedy(cMax));
    % if a column was used already by a different j, there will be a 0
    while nextCol == 0
        ptrGreedy(cMax) = ptrGreedy(cMax) + 1;
        nextCol = orderGreedy{cMax}(ptrGreedy(cMax));
    end
    
    % add to active set
    I2(cI) = nextCol;

    % remove nextCol from all orderGreedy queues
    for cJ = 1:rows
        orderGreedy{cJ}(orderCols{cJ}(nextCol)) = 0;
    end
    
    % update {H*u}_j, and increment pointer to queue j
    runningTotals = runningTotals + D*H(:,ordered(greedyStart-1+nextCol));
    ptrGreedy(cMax) = ptrGreedy(cMax) + 1;
end

% concatenate the trivially determined columns with the greedy heuristic.
I = [I ordered(I2 + greedyStart - 1)];
end