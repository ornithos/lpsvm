function [ model ] = solveModel(X, y, u, kernel)
    tol = 1e-5;
    I = find(u > tol);
    if isempty(I) 
        error('No ui''s found s.t. ui > 0'); 
    end
    
    model.i = I;
    model.x = X(:, I);
    model.y = y(I);
    model.u = u(I);
    vec = model.u .* model.y;
  
    K = kkMat(X, model.x, model.i, kernel); %kkMat(X, model.x, kernel);
    
    model.v = sqrt(vec*K*vec');
end

% Structure
%model{i}.v normalization
%model{i}.y y SV
%model{i}.x SV
%model{i}.u multipliers

