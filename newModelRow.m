function [ row ] = newModelRow(X, y, model, nModel, kernel)
    row =  y .* modelOut(X, model{nModel}, kernel);
end
