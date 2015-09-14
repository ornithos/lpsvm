function out = createIndexVector(n, P, sorted, skipIdx)
% CreateIndexVector(n, P)
% Create a length-n zero vector (logical)
% with 1 (true) for every index in P. This is required for speedup in
% active set routine - set indexing is taking ~ 25% time of linprog!

if ~exist('skipIdx','var')
    skipIdx=[];
else
    % bit of a hack - probably not fastest way, but it's neat.
    elmnts = find(skipIdx);
    P = elmnts(P);
end

if ~sorted
    P = sort(P);
end

% actually create index
out = false(n,1);
setLength = numel(P);
for i = 1:setLength
    out(P(i)) = true;
end

end

