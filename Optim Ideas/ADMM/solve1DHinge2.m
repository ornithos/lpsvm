function [x,z,u] = solve1DHinge2(h, hn2, p, v, rho, tol, maxIter, z0, u0, eta)
%solve1DHinge Using proximal methods/ADMM to solve LPSVM loss function for
%    n = 1 example.
%   Use unconstrained proximal update for x, simplex projection for z
%   v2 -- added Nesterov Acceleration
%         CONVERGES TO (SLIGHTLY) SUBOPTIMAL ANSWER

%   Arguments:
%   h          -   (Vector) vector from data/constraint matrix. Should
%                   already be augmented with a 1 and multiplied by nu_inv.
%   hn2        -   (Scalar) norm squared of h
%   p          -   (Scalar) length of h minus 1.
%   v          -   (Vector) Current value of z - u in outer loop.
%   rho        -   (Scalar) quadratic penalty constant in augm. lagrangian
%   tol        -   (Scalar) terminate when optimal to this tolerance
%   maxIter    -   (Scalar) maximum number of iterations in ADMM loop
%   z0         -   (Vector) warm start z
%   u0         -   (Vector) warm start u

verbose = true;
cc = [0 Inf];
alpha = [1 1];
restarts = 0;
x = z0;

z = z0;
zHat = z0;
zOld = z0;

u = u0;
uHat = u0;
uOld = u0;

c = [1; zeros(p,1)];

% obj = @(x)(max(0,h'*x) - c'*x + (rho/(2))*norm(x-v)^2);
% opts = optimoptions('fmincon','Algorithm','sqp','TolX', 1e-10, 'Display','None');
% opt = fmincon(obj,z0,[],[],[0,ones(1,p)],1,[-Inf;zeros(p,1)], [],[],opts);


for k = 1:maxIter
    t = (rho.*(v+zHat-uHat) + c)'*h./hn2;
    if t > 1
        t = 1;
    elseif t < 0
        t = 0;
    end

    x      = 0.5*(v+zHat-uHat) - (-c + t.*h)./(2*rho);
    z      = x+uHat;
    z(2:end) = projsplx(z(2:end));
    u      = uHat + x - z;

    % Nesterov ================
    uMu = u - uHat; zMz = zHat - z;
    cc(1) = uMu'*uMu + zMz'*zMz;
    if cc(1) < eta*cc(2)
        alpha(1) = 0.5*(1+sqrt(1+4*alpha(2)^2));
        a        = (alpha(2)-1)/alpha(1);
        zHat     = (1 + a)*z -a*zOld;
        uHat     = (1 + a)*u -a*uOld;
    else
        alpha(1) = 1;
        zHat     = zOld;
        uHat     = uOld;
        cc(1)     = cc(2)/eta;
        restarts  = restarts + 1;
    end
    
        
    cc(2) = cc(1);
    alpha(2) = alpha(1);
    zOld = z;
    uOld = u;
    % =========================
    
    history.objval(k)  = max(h'*x,0) - x(1) +(rho/2).*norm(x-v)^2;
    history.r_norm(k)  = norm(x-z);
    history.s_norm(k)  = sqrt(p+1)*norm(-2*rho*(z - zOld));   % z is n times longer

    history.eps_pri(k) = sqrt(p+1)*tol + tol*max(norm(x), sqrt(p+1)*norm(-z));
    history.eps_dual(k)= sqrt(p+1)*tol + tol*norm(2*rho*u);

    if verbose
        fprintf('%3d\t%10.6f\t%10.6f\t%10.6f\t%10.6f\t%10.6f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end
    
    zOld = z;
end

% if abs(obj(x) - obj(opt)) > 1e-3
%     stophere = 0;
% end
end

