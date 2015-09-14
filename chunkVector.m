function chunks = chunkVector(n, cMax)
    % chunk a series 1:n into smaller contiguous series of max length cMax
        cMax          = floor(cMax);
        chunkN        = int32(ceil(n/cMax));
        chunkCut      = [0 repmat(floor(n/chunkN), 1, chunkN)];
        chunkCut      = cumsum(chunkCut);
        chunkCut      = min(chunkCut, n);
        delchk        = [];
        for i = 2:length(chunkCut)
            if chunkCut(i) == chunkCut(i-1)
                delchk = [delchk i];
            end
        end
        chunkCut(delchk) = [];
        
        chunks = arrayfun(@(n) (chunkCut(n)+1):chunkCut(n+1), ...
            1:numel(chunkCut)-1, 'UniformOutput', 0);
end

