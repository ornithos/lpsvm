function addP = pickIndependentCols(H, P, vAmt, NN, pctProj)
    % Selects a subset based on the successive largest vector by partial 
    % (Gram-Schmidt) orthonormalisation. pctProj is a parameter in (0,1]
    % to select how 'strong' the successive orthonormalisation should be.
    % vAmt is the size of the violation.
    
    % Successive GS Orthogonalisations stored in matrix GS
    GS = H(:,P);
    L2orig = sum(GS.^2, 1);
    if(NN > length(P)); NN = length(P); end;
    if size(vAmt,1) > 1; vAmt = vAmt'; end;
    remaining = 1:length(P);
    addP = zeros(1,NN);
    
    % include the column of largest violation first.
    % on subsequent iterations, choose the orthonormalised column
    % that retains the highest L2 norm
    
    cI = 1;     % current iteration
    cP = 1;     % current column. P is initially ordered by violation.
    L2max = sum(GS(:,cP).^2);
    while cI <= NN
        % perform projection with current vector
        vv = GS(:,cP);
        proj = bsxfun(@times, vv'*GS, vv./L2max);
        GS = GS - pctProj.*proj;
        
        % add current column to active set and remove from remaining.
        addP(cI) = remaining(cP);
        GS(:,cP) = [];
        L2orig(cP) = [];
        vAmt(cP) = [];
        remaining(cP) = [];
        
        % move forward one iteration by selecting the largest remaining
        % col. This is weighted by the violation amount.
        cI = cI+1;
        L2 = sum(GS.^2, 1);
        [~, cP] = max(vAmt .* L2./L2orig);
        L2max = L2(cP);
    end
    addP = P(addP);
end