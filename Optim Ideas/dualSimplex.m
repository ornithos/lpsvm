function x = dualSimplex(c,A,b,eps,x0)

% Solves:- Max c^Tx subject to Ax = b, x >= 0
% using the dual simplex method
% We are given a nondegenerate basic solution x0
% x = dual_simplex(c,A,b,eps,x0)
% eps is a suitable tolerance, say 1e-3
% The method implements the dual simplex method in Box 10.1 on page 155
% of Chvatal

% Kartik's MATLAB code for MA/OR/IE 505
% February 20, 2006
% Last modified: April 13th, 2006

[m,n] = size(A);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 0:- Initialize B and N and also x_B and s_N

B = find(x0);
N = find(ones(n,1) - abs(sign(x0)));
xB = x0(B);
y = A(:,B)'\c(B);
sN = c(N)-A(:,N)'*y;

iter = 0;
while 1 == 1,
   iter = iter + 1;

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Step 1:- Check for optimality
   % Else, find the leaving basic variable is x_{B(i)}
   
   [xBmin,i] = min(xB);
   if xBmin >= -eps,
       fprintf('We are done\n');
       fprintf('Number of iterations is %d\n',iter);
       x = zeros(n,1);
       x(B) = xB;
       fprintf('Optimal objective value is %f\n',c'*x);
       return;
   end;
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Step 2:- Update the pivot row
  
   e = zeros(m,1);
   e(i) = 1;
   e = e(:);
   v = A(:,B)'\e;
   w = A(:,N)'*v;
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Step 3 :- Find the entering nonbasic variable x_{N(j)}
   
   zz = find(w < -eps);
   if (isempty(zz))
     error('System is infeasible\n');
   end
     
   [t,ii] = min(sN(zz)./w(zz));
   j = zz(ii(1));
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Step 4 :- Solve Bd = a_{N(j)} for d
   
   d = A(:,B)\A(:,N(j));
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Step 5 :- Update the basis and the basic solution x_B
   % Also update s_N
   
   temp = B(i);
   B(i) = N(j);
   N(j) = temp;

   
   theta = xB(i)/w(j);
   xB = xB - theta*d;
   xB(i) = theta;
   
   sN = sN -t*w;
   sN(j) = -t;
   
   
end; % while   
   