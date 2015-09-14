function [K] = kkMat(X, mX, mI, kernel, allcol)
    % Returns kernel Matrix between mX and X
    % by default, returns a square matrix size of mI,
    % allcol returns a rectangular matrix, where all
    % columns of X are returned.
    
    % There's a problem with persistence of the kernel Cache. IF ERROR WITH
    % KKMAT FUNCTION, SIMPLY HIT 'clear kernelCache', and this should resolve.

    if ~exist('allcol', 'var')
        allcol = false;
    end
    
    [~, N]     = size(mX);
    
    if(~allcol)
        K      = zeros(N, N);
    else
        K      = zeros(N, size(X,2));
    end
    
    for i = 1:N,
        kv  = kernelCache(1, mI(i), []); % find
        if(isempty(kv))
             kv  = kernel(full(mX(:,i)), X);
             kernelCache(2, mI(i), kv); % save
        end
        
        % Return just requested columns or all columns?
        if(~allcol)
            K(i,:) = kv(mI);
        else
            K(i,:) = kv;
        end
    end
end
