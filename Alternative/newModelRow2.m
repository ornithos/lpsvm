function [ out ] = newModelRow(X, y, model, nModel, kernelPar)
    m = model{nModel};
    row = m.u * ...
        bsxfun(@times, m.y'./m.v', ...
        kernel_mat_eq(X, m.x, kernelPar));
    out =  y .* row;
end