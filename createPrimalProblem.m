function [f, A, b, Aeq, beq, lb, ub] = createPrimalProblem(H, D)
%% AB for DEBUG USE ONLY %%%%%%%%%
% Analagous to the createProblem function but for primal.
H = H';
[dim, N] = size(H);

f = [zeros(1, N) D*ones(1,dim) -1];  % Objective function = [a, xi, rho]

A   = [-H, -eye(dim) ones(dim,1)]; % rho - Ha - xi <= 0
b   = zeros(dim, 1);
Aeq = [ones(1, N) zeros(1,dim+1)]; % sum a = 1
beq = 1;
lb = [zeros(1, dim+N) -Inf]; % a, xi >= 0
ub = Inf(1, dim+N+1); % unconstrained ub

end