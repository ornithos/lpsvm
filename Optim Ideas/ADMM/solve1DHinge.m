function [x,z,u] = solve1DHinge(h, hn2, p, v, rho, tol, maxIter, z0, u0)
%solve1DHinge Using proximal methods/ADMM to solve LPSVM loss function for
%    n = 1 example.
%   Use unconstrained proximal update for x, simplex projection for z

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

verbose = false;
x = z0;
z = z0;
zOld = z0;
u = u0;
c = [1; zeros(p,1)];
obj = @(x_var)( pos(h'*x_var) - c'*x_var + rho/2*sum_square(x_var - v) );
% obj = @(x)(max(0,h'*x) - c'*x + (rho/(2))*norm(x-v)^2);
% opts = optimoptions('fmincon','Algorithm','sqp','TolX', 1e-10, 'Display','None');
% opt = fmincon(obj,z0,[],[],[0,ones(1,p)],1,[-Inf;zeros(p,1)], [],[],opts);


for k = 1:maxIter
    t = (rho.*(v+z-u) + c)'*h./hn2;
    if t > 1
        t = 1;
    elseif t < 0
        t = 0;
    end

    x      = 0.5*(v+z-u) - (-c + t.*h)./(2*rho);
%     cvx_begin quiet
%         variable x_var(p+1)
%         minimize ( 0.5*pos(h'*x_var) - 0.5*c'*x_var + rho/2*sum_square(x_var - 0.5*(v+z-u)) )
%     cvx_end
%     if norm(x-x_var) > 1e-6
%         stophere = norm(x-x_var);
%         stophere = 0;
%     end
%     if any(isnan(x_var))
%         stophere = 0;
%     end
%     x = x_var;
    z      = x+u;
    z(2:end) = projsplx(z(2:end));
    u      = u + x - z;

    history.objval(k)  = obj(x);
    history.r_norm(k)  = norm(x-z);
    history.s_norm(k)  = sqrt(p+1)*norm(-2*rho*(z - zOld));   % z is n times longer

    history.eps_pri(k) = sqrt(p+1)*tol + tol*max(norm(x), sqrt(p+1)*norm(-z));
    history.eps_dual(k)= sqrt(p+1)*tol + tol*norm(2*rho*u);

    if verbose
        fprintf('%3d\t%1.7f\t%1.7f\t%1.7f\t%1.7f\t%1.7f\n', k, ...
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



%     subject to
%         h'*[0;1;1] == 1;
%         x_var >= [-1000; 0; 0];
