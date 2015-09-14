randn('state', 0);
rand('state', 0);

n = 300;  % dimension of x
m = 200;  % number of equality constraints

c  = rand(n,1) + 0.5;    % create nonnegative price vector with mean 1
x0 = abs(randn(n,1));    % create random solution vector

A = abs(randn(m,n));     % create random, nonnegative matrix A
b = A*x0;

[x history] = linprogRplus(c, A, b, 1.0, 1.0);


%% Consensus
K = length(history.objval);

h = figure;
plot(1:K, history.objval, 'k', 'MarkerSize', 10, 'LineWidth', 2);
ylabel('f(x^k) + g(z^k)'); xlabel('iter (k)');

g = figure;
subplot(2,1,1);
semilogy(1:K, max(1e-8, history.r_norm), 'k', ...
    1:K, history.eps_pri, 'k--',  'LineWidth', 2);
ylabel('||r||_2');

subplot(2,1,2);
semilogy(1:K, max(1e-8, history.s_norm), 'k', ...
    1:K, history.eps_dual, 'k--', 'LineWidth', 2);
ylabel('||s||_2'); xlabel('iter (k)');


[x history] = linprogDist(c, A, b, 0.6, 4);

%% General Consensus
n = 200;  % dimension of x
m = 60;  % number of equality constraints

c  = rand(n,1) - 0.5;    % create nonnegative price vector with mean 1
x0 = abs(randn(n,1));    % create random solution vector

A = abs(randn(m,n));     % create random, nonnegative matrix A
b = A*x0;


% Create Inequality Constrained Problem
slack = exprnd(0.2, m, 1);
slack(rand(m,1)>slack*3) = 0;
bineq = b + slack/10;

npar = 4;
d    = m/npar;
Apar = cell(npar,1); bpar = cell(npar,1); cpar = cell(npar,1);
for i = 1:npar
    Apar{i} = [eye(d), A(((i-1)*d+1):(d*i),:)];
    bpar{i} = bineq(((i-1)*d+1):(d*i));
    cpar{i} = [0*ones(d,1); c];
end
[x history] = linprogDist2(cpar, Apar, bpar, d, 0.2, [], 350);



%% (Variable Length) General Consensus
n = 200;  % dimension of x
m = 60;  % number of equality constraints

c  = rand(n,1) - 0.5;    % create nonnegative price vector with mean 1
x0 = abs(randn(n,1));    % create random solution vector

A = abs(randn(m,n));     % create random, nonnegative matrix A
b = A*x0;


% Create Inequality Constrained Problem
slack = exprnd(0.2, m, 1);
slack(rand(m,1)>slack*3) = 0;
bineq = b + slack/10;

npar = 4;
d    = m/npar;
Apar = cell(npar,1); bpar = cell(npar,1); cpar = cell(npar,1); lb = cell(npar,1);
for i = 1:npar
    Apar{i} = [eye(d), A(((i-1)*d+1):(d*i),:)];
    bpar{i} = bineq(((i-1)*d+1):(d*i));
    cpar{i} = [0*ones(d,1); c];
    lb{i} = zeros(n+d,1);
end
[x history] = linprogDist3(cpar, Apar, bpar, lb, d, 0.2, [], 350);


% Looking at LPSVM problem
% for i = 1:4
%     Aeq{i} = [eye(np(i)), -eye(np(i)), -ones(np(i),1), par{i}.H'];
%     c{i}   = [D2*ones(np(i),1); zeros(np(i),1); -par{i}.n; zeros(p,1)]; 
%     beq{i} = [zeros(np(i),1);1];
%     lb{i}  = [zeros(2*np(i),1);-Inf; zeros(p,1)];
% end
% assignin('base','lpA',Aeq);
% assignin('base','lpb',beq);
% assignin('base','lpc',c);
% assignin('base','lplb',lb);
for i = 1:4
    lpA{i} = [lpA{i}; [zeros(1,size(lpA{i},2)-2), ones(1,2)]];
end
[x2 history] = linprogDist3(lpc, lpA, lpb, lplb, -3, 1, [], 350);
