function [ out, set ] = countSV( model, Nmodels, a )
% self-explanatory - count the number of support vectors currently in
% model. Only learners for which a_i > 0 are included.

tol = 1e-4;

SVs = [];
for i=1:Nmodels,
    if(a(i) > tol), % Only models that count
        SVs = union(SVs, model{i}.i);
    end
end

out = length(SVs);
set = SVs;
end

