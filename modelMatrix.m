function [ H ] = initModelMatrix(X, y, model, kernel, H)
    nModels  = length(model);
   
    for i=1:nModels, % one row per model
        H(i,:)     = y .* modelOut(X, model{i}, kernel);
    end
end

function out = modelOut(X, model, kernel)
    [~, N] = size(X);
    out    = zeros(1, N);
    for i=1:N,
        out(i) = sum(model.u .* model.y .* kernel(X(:,i), model.x))/model.v;
    end
end

% Structure
%model{i}.v normalization
%model{i}.y y SV
%model{i}.x SV
%model{i}.u multipliers

