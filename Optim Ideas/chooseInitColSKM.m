
function I = chooseInitColSKM(H, NN, D, maxIter)

% Perform Spherical KMeans on the remaining columns
% in order to find the most representative subset of the columns
% under scale invariance.

% CURRENTLY IGNORING THE FACT THAT MAY BE CHOOSING MANY SIMILAR TO THE WARM
% START COLUMNS SPECIFIED BEFORE THIS FUNCTION...

if nargin < 4
    maxIter = 50;
end

[nrow, ncol] = size(H);
% Bail if too few columns relative to NN to be worth doing.
if NN > ncol/3;
    I = randperm(ncol, NN)';
    return
end

% Generate initial mu on surace of unit sphere
mu = H(:,randperm(ncol, NN))';
mu = bsxfun(@rdivide, mu, sqrt(sum(mu.^2, 2)));

% previous assignment for stopping criterion
prev = zeros(1,ncol);

for i = 1:maxIter
    % assign new points to cluster
    asgn = mu*H;
    [~,y] = max(asgn, [], 1);

    % stopping criterion
    if all(prev == y)
        break
    end
    prev = y;
    
    % compute unnormalised centroid updates
    for j = 1:NN
        mu(j,:) = mean(H(:,y == j), 2)';
        if isnan(mu(j,1))
            fprintf('No points in cluster %d', j);
            mu(j,:) = randperm(ncol, 1);
        end
    end
    
    % normalise
    mu = bsxfun(@rdivide, mu, sqrt(sum(mu.^2, 2)));
end
    
% Find closest point to each centroid
[~,I] = max(asgn, [], 2);

% In general we will encounter the case where an index is assigned to more
% than one centroid. (by properties of kmeans) This is handled below.

I = sort(I)';

% indices to unique values in column 3
[I2, ind] = unique(I);

if ~length(I) == length(I2)
    dupInd = setdiff(1:NN, ind);
    remaining = setdiff(1:NN, I2);
    rp = randperm(remaining);
    
    % This should all be done with a set/binary tree object but MATLAB :X
    pool = [];
    rpNum = 1;
    for j = 1:length(dupInd)
        el = I(dupInd(j));
        if ~ismember(el, pool)
            % Not seen yet, so no need to remove this one
            pool = [pool el];
        else
            % overwrite I with non-used vector
            I(dupInd(j)) = rp(rpNum);
            rpNum = rpNum+1;
        end
    end
end
end