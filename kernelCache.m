function [ out ] = kernelCache( op, rowpos, row )

persistent cache cacheSize;

% clear functions (To clear the cache)
if(op == -1)
    cacheSize = 55933;
    cache     = cell(1, cacheSize);
    out = true;
    return
end

% Create cache if none
if isempty(cache) %
    cacheSize = 55933;
%    cacheSize = 26893;
    cache     = cell(1, cacheSize);
end

if(op == 1) % find
    out = find(rowpos, cache, cacheSize);
else % save
    [cache] = save(rowpos, row, cache, cacheSize);
    out = [];
end


end

% functions to control the cache
function [out] = find(rowpos, cache, cacheSize)
    pp = mod(rowpos, cacheSize)+1;
    out  = [];
    if(isfield(cache{pp}, 'pos') && (cache{pp}.pos == rowpos))
         out = cache{pp}.row;
    end
end

% Pure hash map. Improve with least used eviction or at least open hashing
% functions to control the cache
function [cache] =  save(rowpos, row, cache, cacheSize)
    pp = mod(rowpos, cacheSize)+1;
    cache{pp}.pos = rowpos;
    cache{pp}.row = row;
end

