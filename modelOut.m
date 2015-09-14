function out = modelOut(X, model, kernel)
    [~, N] = size(X);
    out    = zeros(1, N);
    
    vv  = model.u .* model.y;
    for i = 1:length(model.i),
        kv  = kernelCache(1, model.i(i), []); % find
        if(isempty(kv))
             kv  = kernel(full(model.x(:,i)), X); % make full the sv to simplify
             kernelCache(2, model.i(i), kv); % save
        end
        out = out + kv*vv(i);
    end
    
    out = out/model.v;
end