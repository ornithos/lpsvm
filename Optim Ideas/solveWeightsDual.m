function dual = solveWeightsDual(H, D, a, rho, xi, fval)
%solveWeightsDual(H, D, a, rho, xi)
%   Given the optimal primal value, solve the KKT system for the optimal
%   dual values.
    tol     = 1e-6;    
    [~,n]   = size(H);
    
    achieve = H'*a - rho;
    
    Up      = xi > tol;
    nUp     = sum(Up);
    
    J       = a > tol;
    nJ      = sum(J);
    
    reqd    = ceil((1-D*nUp)/D);
    avail   = nJ;
    if reqd >  avail
        error(['KKT dual solve failure - more indices to be determined than ', ...
            'constraints (%d > %d)'], reqd, avail);
    end

    u       = zeros(n,1);
    u(Up)   = D;
    
    if reqd >= 1
        % Slim down the possible candidates for u_i > 0
        [~,remain]  = sort(achieve, 'ascend');
        remain  = remain(1:(nUp+avail));
        remain  = setdiff(remain, find(Up));
    
        M       = [H(J,remain); ones(1, length(remain))];
        b       = [fval*ones(avail,1) - D*sum(H(J,Up),2); 1 - nUp*D];
        freeval = lsqnonneg(M,b);

        u(remain) = freeval;
    end

    dual.u     = u;
    dual.beta  = fval;

end

