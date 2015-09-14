function X = cholPrimalLP(H, rho)
%cholPrimalLP Computes the Cholesky Decomposition of the system of
% equations arising from the primal LP in ADMM.
%
% The purpose of this function is to reduce computational complexity by
% obviating the large matrix multiplication A^T A, and the resulting
% Cholesky decomposition chol(A^T A). This has an almost entirely analytic
% form and the function does little more than allocate n(n+1)/2 elements
% of a matrix and calculate H*H^T (<0.25 size of A^T A) and its Cholesky 
% Decomposition.

% TO DO:
% GIVEN THAT > 90% of CHOL MATRIX IS EMPTY.. RETURN ONLY LEADING DIAGONALS
% OF IDENTITIES, AND THE BOTTOM DENSE SECTION. INITIAL CHOL*VEC CALCS CAN
% THEN ALSO BE SPARSIFIED.

%% Compute basic quantities
[p, n] = size(H);
rho2 = rho^2;
rho4 = rho2^2;

% All scalars
a    = 1/sqrt(1+rho2);
b    = 1/sqrt(rho2*(2 + 3*rho2 + rho4));
c    = sqrt((1+rho2)/(2*rho2 + rho4));

r    = sqrt((n*rho2)/(2 + rho2) + rho2);
s    = -(n*rho2)/(r*(2 + rho2));
w    = sqrt(r^2 - s^2);

f1   = c*(c - b)/r;
f3   = (s + r)*f1/w;

d    = 1/(2 + rho2);
u    = (2 + rho2)/(rho2*(2 + rho2 + 2*n));


%% Compute Schur Complement Q
HHT   = H*H';
rsH   = sum(H,2);            % row sum of H
Q     = (1 - 2*d).*HHT - 2*(4*d*u*(d-1) + u)*(rsH*rsH') + ones(p,p) + rho2*eye(p);
Qchol = chol(Q)';


%% Compute Required Cholesky Decomposition
X = zeros(2*n+2+p);
idx = [1, n;
       n+1, 2*n;
       2*n+1, 2*n+1;
       2*n+2, 2*n+2;
       2*n+3, 2*n+2+p];
   
X(idx(1,1):idx(1,2), idx(1,1):idx(1,2))   = (1/a).* eye(n);

X(idx(2,1):idx(2,2), idx(1,1):idx(1,2))   = -a .* eye(n);
X(idx(2,1):idx(2,2), idx(2,1):idx(2,2))   = (1/c) .* eye(n);

X(idx(3,1):idx(3,2), idx(1,1):idx(1,2))   = -a.*ones(1, n);
X(idx(3,1):idx(3,2), idx(2,1):idx(2,2))   = (c-b).*ones(1, n);
X(idx(3,1):idx(3,2), idx(3,1):idx(3,2))   = r;

X(idx(4,1):idx(4,2), idx(1,1):idx(1,2))   = a.*ones(1, n);
X(idx(4,1):idx(4,2), idx(2,1):idx(2,2))   = -(c-b).*ones(1, n);
X(idx(4,1):idx(4,2), idx(3,1):idx(4,2))   = [s, w];

X(idx(5,1):idx(5,2), idx(1,1):idx(1,2))   = a.*H;
X(idx(5,1):idx(5,2), idx(2,1):idx(2,2))   = (b-c).*H;
X(idx(5,1):idx(5,2), idx(3,1):idx(3,2))   = (2*f1 - 1/r)*rsH;
X(idx(5,1):idx(5,2), idx(4,1):idx(4,2))   = ((s+r)/(r*w) - 2*f3)*rsH;
X(idx(5,1):idx(5,2), idx(5,1):idx(5,2))   = Qchol;

end

