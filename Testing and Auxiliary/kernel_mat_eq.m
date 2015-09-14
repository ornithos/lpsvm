function K = kernel_mat_eq(x, y, sigma)
% Construct Exponentiated Quadratic rbf kernel **MATRIX**
assert(size(x,1) == size(y,1));

diffSq = bsxfun(@plus, sum(x.^2, 1), sum(y.^2, 1)') - 2*y'*x;
K = exp(-diffSq./sigma);
end

