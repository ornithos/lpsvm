function [f, Aeq, beq] = createStandardFormProblem(H, D)
%% Create objects s.t. LP is of form 
%  min f^T x
%  s.t. Ax = b
%  x >= 0

[m, n] = size(H);

f = [zeros(1,2*n+m) 1 -1];
Aeq = [H eye(m) zeros(m,n) -ones(m,1) ones(m,1); ...
    eye(n) zeros(n,m) eye(n) zeros(n,2); ...
    ones(1,n) zeros(1,m+n+2)];
beq = [zeros(m,1); ones(n,1).*D; 1];

end

