function [outA, outB] = lsBoundarySimplex(a, b)
% lsBoundarySimplex(a, b) Shrink/Grow a line segment described by [a,b]
% such that the start and end point are on the boundary of the simplex
% Warning: fminbnd can return Infinite if it fails to find finite values in
% initial iterates. Hence the while loops.

    a      = projsplx(a);  % ensure prototypes are *in* the simplex
    b      = projsplx(b);  % so the induced direction 'a' sums to 0.
    d      = b - a;
    
    bnd     = 1/max(abs(d));
    fv      = Inf;
    while isinf(fv)
        [t,fv]  = fminbnd(@(t) inHypercube(a,d,t,-1), -bnd, bnd);
        bnd = bnd/2;
    end
    outA    = a+t.*d;
    
    bnd     = 1/max(abs(d));
    fv      = Inf;
    while isinf(fv)
        [t,fv]  = fminbnd(@(t) inHypercube(a,d,t,+1), -bnd, bnd);
        bnd = bnd/2;
    end
    outB    = a+t.*d;
end


function out = inHypercube(a, d, t, dir)
% dir should be +1/-1 depending on direction of optimisation
penalty = Inf;
if any(a+t.*d > 1)
    out = dir.*t +penalty;
elseif any(a+t.*d < 0)
    out = dir.*t +penalty;
else
    out = dir.*t;
end
end