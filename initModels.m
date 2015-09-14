function [ model ] = initModels(X, y, initm, kernel, sinit, model)
% initModels. Initialise first initm rows of the matrix H.
% choose sinit % support vectors uniformly at random (without replacement)

    [~, N] = size(X);
    initN  = floor(sinit*N);
    if initN < 1
        error('sinit too small for size of dataset; 0 cols selected..');
    end
    
    for i=1:initm,
        I = randperm(N,initN);
        model{i}.i = I;
        model{i}.x = X(:, I);
        model{i}.y = y(I);
        model{i}.u = ones(1, length(I))/length(I); % 1/M
    
        vec = model{i}.u .* model{i}.y;
        K = kkMat(X, model{i}.x, model{i}.i, kernel);
        %K = kernMat(model{i}.x, kernel);
        
        model{i}.v = sqrt(vec*K*vec');
    end
end

% Structure
%model{i}.v normalization
%model{i}.y y SV
%model{i}.x SV
%model{i}.u multipliers
%model{i}.i index of SV


% function [K] = kernMat(X, kernel)
%     [~, N] = size(X);
%     K      = zeros(N, N);
%     for i=1:N,
%         K(i, :) = kernel(X(:,i), X);
%     end
% end

