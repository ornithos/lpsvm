function PlotLPSVM(model, X, y, typeContour)
% Plot Decision Boundary of trained SVM on input space

[m, n] = size(X);

% Assume X has row-wise elements.
if m < n
    X = X';
end

if min(m,n)>2
    if(min(m,n) == 3 && all(X(:,1)==1))  
        % Is there an intercept term in the data?
        plotX = X(:,2:3);
        bIntercept = true;
    else
        display('X must be 2 dimensional - cannot plot input space otherwise.');
        return 
    end
else
    % No intercept term in the data
    plotX = X;
    bIntercept = false;
end

% Indices of each class
[ym, yn] = size(y);
if ym < yn; y = y'; end;
if size(y,2) ~= 1
    display('y must be a single column vector')
    return
end
cls = unique(y);
if size(cls,1) > 2
    display('y can have at most 2 distinct values.')
    return
end
yIdx = logical(bsxfun(@eq, y, cls'));

%% Setup
svInd = model.IsSupportVector;
if issorted(size(svInd)); svInd = svInd'; end;

gridSize = 200;
x1 = linspace(min(plotX(:,1)), max(plotX(:,1)), gridSize);
x2 = linspace(min(plotX(:,2)), max(plotX(:,2)), gridSize);
[X1,X2] = meshgrid(x1,x2);

% SCORE GRID
gridX = [X1(:) X2(:)];
if bIntercept; gridX = [ones(gridSize^2,1) gridX]; end;
if isa(model, 'ClassificationSVM')
    score = predict(model, gridX);
else
    score = predictLPSVM(model, gridX', false);
    %score = sign(evalModel(gridX',model.model, model.a, ...
    %    @(x,y) gauss(x,y,model.KernParams)));
end
scoreGrid = reshape(score,gridSize,gridSize);

%% Plot
figure
markerTypes = cell(2,1);
markerTypes{1} = 'k.'; markerTypes{2} = 'k+';

% Plot Decision Areas
if typeContour
    contour(X1,X2,scoreGrid)
    colorbar;
else
    gscatter(X1(:),X2(:),score,  [0.855 0.875 0.161; 0.957 0.965 0.812]);
end
hold on
    
% Plot Data
for cC = 1:size(cls,1)
    ix = yIdx(:,cC);
    plot(plotX(ix,1),plotX(ix,2),markerTypes{cC})
    plot(plotX(ix & svInd,1),plotX(ix & svInd,2),'ro','MarkerSize',10)
end
title('{\bf SVM Decision Boundary}')
xlabel('dim 1')
ylabel('dim 2')
legend('Observation','Support Vector')
hold off


end

