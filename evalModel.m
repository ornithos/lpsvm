function [ out ] = evalModel( X, model, a, kernel)
    [~, N] = size(X);
    nModels  = length(model);
    H        = zeros(nModels, N);
    
    for i=1:nModels, % one row per model
        H(i,:)     = modelOut(X, model{i}, kernel);
    end
    out = a'*H;
end

function out = modelOut(X, model, kernel)
    [~, N] = size(X);
    out    = zeros(1, N);
    for i=1:N,
        out(i) =  sum(model.u .* model.y .* kernel(X(:,i), model.x))/model.v;
    end
end