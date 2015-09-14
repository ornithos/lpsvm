function [x, y] = LPLSProx1(f, A, b, rho, maxIter, tol)
%LPLS1 Least Squares Solution via Proximal Methods
%   
if size(f,2)>1; f = f'; end;
if size(b,2)>1; b = b'; end;

[m, n] = size(A);
xOld = ones(n,1)/n;

B = [rho.*eye(n) A'; A zeros(m)];

for i = 1:maxIter
    z = [rho.*xOld - f; b];
    xNew = lsqnonneg(B,z);
    
    nu = xNew((n+1):end);
    xNew = xNew(1:n);
    if all(A'*nu <= f) && abs(b'*nu - f'*xNew) < tol*n && sum(abs(A*xNew - b)) < tol*n
        break
    end
    
    xOld = xNew;
end

x = xNew;
y = nu;
end
