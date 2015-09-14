function [xave, history] = linear_svm3(A, z, u, rho, alpha, tol)
% linear_svm3   
%
% -- Using the hinge loss ADMM solver instead of CVX
% Adapted from Boyd et al. Solves the LPSVM problem on a one-at-a-time
% schedule.
%
% Solves the following problem via ADMM:
%
%   minimize   (h'*x)_+ - c'*x + (rho/2)||x-z+u||_2^2
%   s.t. a \in simplex
%   for each i in [n].
%
% This function implements a *distributed* SVM that runs its updates
% serially.
%
% The solution is returned in the vector x = (primal.rho,primal.a).

t_start = tic;

QUIET    = 0;
MAX_ITER = 1000;
ABSTOL   = tol;
RELTOL   = tol*10;

[p, n] = size(A);

% group samples together
for i = 1:n,
    tmp{i} = A(:,i)';
end
MnormSq = sum(A.*A);
A = tmp;


x = zeros(p,n);
z = repmat(z,1,n);
% z = zeros(p,n);
% u = zeros(p,n);
c = [1; zeros(p-1,1)];


if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

count=0;
for k = 1:MAX_ITER
	% x-update
    for i = 1:n,

%         cvx_begin quiet
%             variable x_var(p)
%             minimize ( pos(A{i}*x_var) - c'*x_var + rho/2*sum_square(x_var - z(:,i) + u(:,i)) )
%             subject to
%                 A{i}*[0;1;1] == 1
%                 x_var >= [-1000; 0; 0]
%         cvx_end
        tmp = solve1DHinge(A{i}',MnormSq(i),p-1,z(:,i)-u(:,i),1,1e-12,40,...
                    z(:,i), u(:,i));
        x(:,i) = tmp;
%         if norm(tmp-x_var) > 1e-8
%             stophere = 1;
%         end
    end
    xave = mean(x,2);

    % z-update with relaxation
    zold = z;
    x_hat = alpha*x +(1-alpha)*zold;
    z = mean( x_hat + u, 2 );
    z = z*ones(1,n);

    % u-update
    u = u + (x_hat - z);

    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(A, x, n);

    history.r_norm(k)  = norm(x - z);
    history.s_norm(k)  = norm(-rho*(z - zold));

    history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
    history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);

    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end
end

if ~QUIET
    toc(t_start);
end
end

function obj = objective(A, x, n)
    obj = hinge_loss(A,x)/n;
end

function val = hinge_loss(A,x)
    val = 0;
    for i = 1:length(A)
        val = val + sum(pos(A{i}*x(:,i))) - x(1,i);
    end
end