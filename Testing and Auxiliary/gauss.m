function [ val ] = gauss( x, y, sigma )
% Construct Exponentiated Quadratic rbf kernel **SCALAR FUNCTION**
% x is the patern, y is the set of SVs
    o  = sum(bsxfun(@minus, y , x).^2, 1); 
    val = exp(-o/sigma);
end
