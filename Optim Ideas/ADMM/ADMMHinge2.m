function [primal, dual, fval, exitflag] = ADMMHinge2(H, D, rho, alpha, ...
                        maxIter, tol, reltol, x0, npar, dualreq, verbose)
% ADMMDistributed: Solve Primal via ADMM by exploiting the simple form of
%  prox((w^T a)_+). We compute only the values of primal.a and primal.rho
%  which avoids the high dimensional consensus optimisation involved for xi
%  and omega in the LP version (ADMMDistributedX). This is at a cost of a
%  separate variable for each datapoint in H. Simplex projection of a is
%  applied in the second function. See Parikh & Boyd - Proximal Operators 
%  ch3 and ch5 for more details.
%
%  While subgradient methods failed, or at least didn't work fast, or
%  didn't choose good descent directions, ||.||^2 gives a smooth approx to
%  the function with the same fixpoints. Further, we have to do prox anyway
%  with ADMM, so this is a natural idea.
%
%   v1
%   -- inner ADMM step: hinge solved under simplex constraint.
%   -- This makes it worse!!
%
%   Arguments:
%   H          -   (Matrix) inequality constraint corr. to Hu <= beta
%   D          -   (Scalar) upper box constraint
%   rho        -   (Scalar) quadratic penalty constant in augm. lagrangian
%%%%   alpha      -   (Scalar) over-relaxation parameter in (1,2)
%   maxIter    -   (Scalar) maximum number of iterations in ADMM loop
%   tol        -   (Scalar) stopping criterion.
%   reltol     -   (Scalar) relative convergence component.
%   x0         -   (Vector) warm start option. (DEPRECATED)
%%%%   npar       -   (Scalar) Number of partitions / nodes to simulate.
%   dualreq    -   (Logical) Recalculate dual variables? Use for precision.
%
%   Outputs:
%   primal     -   Primal variables. Feasible but not necessarily optimal.
%                  Margin is approximately correct, but H'a + xi may be less
%                  than margin for a few datapoints.
%   dual       -   Dual Variables. Again, are only approximate, and some Hu
%                  may exceed beta. Null is dualreq is false.
%   exitflag   -   1  = converged, 0 = number of iterations exceeded,
%                  -1 = iterations exceeded / infeasibility > 10*tol.

%% set-up
% ensuring correct orientation of vectors
if any(size(D))>1; error('D must be scalar'); end;
    

[p, n]         = size(H);
nu_inv         = D*n;

% augment H with bias
M              = [ones(1,n); -H].*nu_inv;
MnormSq        = sum(M.*M);

% Variables
X              = zeros(size(M));
U              = zeros(size(M));
z              = [0; ones(p,1)./p];
zOld           = [0; ones(p,1)./p];
c              = [1; zeros(p,1)];
ZmUpC          = repmat(z+c, 1, n);


saveZ = zeros(p+2+p+2+p+2,maxIter);

%% Main ADMM Section
% following code adapted from S. Boyd, N. Parikh, E. Chu, B. Peleato, 
% and J. Eckstein: https://web.stanford.edu/~boyd/papers/admm/linprog/linprog.html

if verbose
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

opts = optimoptions('fmincon','Algorithm','sqp','TolX', 1e-10, 'Display','None');
obj = @(x,h,v)(max(0,h'*x) - c'*x + (rho/(2))*norm(x-v)^2);
for k = 1:maxIter

    % === x-update ====
    for i = 1:n
        tmp1 = solve1DHinge(M(:,i),MnormSq(:,i),p,z-U(:,i),1,1e-19,40,...
                    [0;ones(p,1)./p], zeros(p+1,1));
%         tmpobj = @(x) obj(x, M(:,i), z-U(:,i));
%         [tmp2,tmpj,tmpx] = fmincon(tmpobj,X(:,i),[],[],[0,ones(1,p)],1,[-Inf;zeros(p,1)],[],[],opts);        
%         if norm(tmp1-tmp2) > 1e-4
%             stophere=0;
%         end
        X(:,i) = tmp1;
    end
    
    if alpha ~= 1
        Xhat = bsxfun(@plus, alpha.*X, (1-alpha).*z);   % relaxation
    else
        Xhat = X;
    end
    
    % === z-update ====
    z        = mean(Xhat+U,2);                      % GATHER (Consensus)
%     z        = mean(Xhat,2);
    % === u-update ====
    XmZ  = bsxfun(@minus, Xhat, z);
    U    = U + XmZ;
    
    %% diagnostics, reporting, termination checks
    history.objval(k)  = sum(max(sum(M.*X,1),0))./n - sum(X(1,:))./n;
    history.r_norm(k)  = norm(XmZ(:));
    history.s_norm(k)  = sqrt(n)*norm(-rho*(z - zOld));   % z is n times longer

    history.eps_pri(k) = sqrt(n)*tol + reltol*max(norm(X(:)), sqrt(n)*norm(-z));
    history.eps_dual(k)= sqrt(n)*tol + reltol*norm(rho*U(:));

    if verbose
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end
    
    % reset for next iteration
    ZmUpC            = bsxfun(@minus, z + c, U);
    zOld             = z;
    saveZ(1:(p+1),k) = [X(:,1)];
    saveZ((p+2):(2*p+3),k) = [NaN; z];
    saveZ((2*p+4):(end-1),k) = [NaN; U(:,1)];
    
end

%% Output
saveZ(end,1:k) = history.objval;
fprintf('num iter: %d, fpval: %1.7f, rho: %1.7f\n', k, ...
    history.objval(k), z(1));

fval       = history.objval(k);
exitflag   = 1;

if k == maxIter
    exitflag = 0;
    if (history.r_norm(k) > 10*history.eps_pri(k) || ...
            history.s_norm(k) > 10*history.eps_dual(k))
        exitflag = -1;
    end
end

% Assign primal/dual variables
primal.a   = z(2:end);
primal.rho = z(1);
primal.xi  = max(primal.rho - H'*primal.a,0);

%% Recalulate Dual Variables using KKT System. While dual variables
%  technically converge, this happens too slowly to be useful. Here we
%  assume that the (i) active set (u_i > 0) has been found, (ii) the
%  bounded vectors are known (u_i == D), and (iii) the optimal beta is
%  known (fval), since the objective val converges quickly. The resulting
%  system is used to calculate the (0 < u_i < D) variables exactly.

if dualreq
    viol    = z(1:n) > 1e-5;
    nm      = sum(viol);

    J       = primal.a > 1e-5;
    nJ      = sum(J);
    reqd    = ceil((1-D*nm)/D);
    avail   = nJ;
    if reqd >  avail
        error(['KKT dual solve failure - more indices to be determined than ', ...
            'constraints (%d > %d)'], reqd, avail);
    end

    u       = zeros(n,1);
    u(viol) = D;

    if reqd >= 1
        gap     = abs(z((n+1):2*n) + z(1:n));
        [~,go]  = sort(gap);
        free    = go(1:avail);
        M       = [H(J,free); ones(1, avail)];
        b       = [-fval*ones(avail,1) - D*sum(H(J,viol),2); 1 - nm*D];
        freeval = lsqnonneg(M,b);

        u(free) = freeval;
    end


    dual.u     = u;
    dual.beta  = -fval;
else
    dual       = [];
end
end


% DEBUG: CHECKING THAT X and Z STEPS MINIMISE AUGM. LAGRANGIAN
%% --- X -----
%     ZmU = bsxfun(@minus, z, U);
% %     obj = @(X, ZmU)(max(0,sum(X.*M)) - X(1,:) + 0.5*sum((X-ZmU).*(X-ZmU)));
%     objSgl = @(X, ZmU,i)(max(0,sum(X.*M(:,i))) - X(1,:) + 0.5*sum((X-ZmU).*(X-ZmU)));
% %     pobj = @(X,ZmU, inc) (fprintf('%s\n',sprintf('%d ', ...
% %         obj(inc, ZmU) - obj(X, ZmU))));
% 
%     for i = 1:n
%         fdff = fminsearch(@(X) objSgl(X, ZmU(:,i),i), X(:,i)) - X(:,i);
%         if max(abs(fdff)) > 1e-8
%             stophere = 0;
%         end
%     end

%% --- Z ----    
%     obj = @(z, X,U)(0.5*norm(repmat(z,n,1)-(X(:)+U(:)))^2);
%     fdff = fmincon(@(z)obj(z,X,U),z,[],[],[0,ones(1,p)],1,[-Inf;zeros(p,1)], ...
%         [],[],optimset('Display','None'));
%     if max(abs(fdff - z)) > 1e-5
%         stophere = 0;
%     end