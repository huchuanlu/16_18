function [model] = train_model(data, label, trainIndex, options)
%%  function [model] = train_model(data, label, trainIndex, option)

[height, width, bands] = size(data);
model = [];
%
param = options.param;
%
winH = options.winH;
winW = options.winW;
dataMatrix = reshape(data, height*width, bands)';
dataMatrix = dataMatrix./(repmat(sqrt(sum(dataMatrix.^2)), [bands 1])+eps);
[height, width, bands] = size(data);
indexMatrix = [1:height*width]';
indexMatrix = reshape(indexMatrix, [height, width]);
indexMatrix(label==0) = 0;
indexMatrix = blkdiag(zeros(winH,winW), indexMatrix, zeros(winH,winW));
%
dataTrain  = [];
labelTrain = [];
for num = 1:length(trainIndex)
    c = floor((trainIndex(num)+height-0.01)/height);
    r = trainIndex(num) - (c-1)*height;
    neighborIndex = indexMatrix(r:r+2*winH, c:c+2*winW);
    neighborIndex = neighborIndex(:);
    neighborIndex(neighborIndex==0) = [];
    dataNeighbor = dataMatrix(:,neighborIndex);
    dataTrain  = [dataTrain, dataNeighbor];
    labelTrain = [labelTrain, label(trainIndex(num))*ones(1,size(dataNeighbor,2))];
end
%
nLabels = max(max(label));
model.dictionary = [];
model.label = [];
for num = 1:nLabels
    dataTemp = dataTrain(:,(labelTrain==num));
    covm = dataTemp*dataTemp';
    pc   = pcacov(covm);
    model.dictionary = [model.dictionary, pc(:,1:param.K)];
    model.label = [model.label, num*ones(1,param.K)];
end
lambda = 0.1/((param.K/5)^1.5);
model.projection = inv(model.dictionary'*model.dictionary+lambda*eye(size(model.dictionary,2)))*model.dictionary';
%
model.svmfeature = [];
model.svmlabel   = [];
for num = 1:length(trainIndex)
    c = floor((trainIndex(num)+height-0.01)/height);
    r = trainIndex(num) - (c-1)*height;
    neighborIndex = indexMatrix(r:r+2*winH, c:c+2*winW);
    neighborIndex = neighborIndex(:);
    neighborIndex(neighborIndex==0) = [];
    dataNeighbor = dataMatrix(:,neighborIndex);
    coeff = model.projection*dataNeighbor;
    coeff = sum(coeff,2);
    coeff = coeff./(norm(coeff)+eps);
    model.svmfeature = [model.svmfeature, coeff];
    model.svmlabel   = [model.svmlabel, label(trainIndex(num))];
end
%
model.svmmodel   = svmtrain(model.svmlabel',model.svmfeature',sprintf('-q -g 1 -t 1 -b 1 -d %d -r 1 -c %f',5,10));
%
model.nLabels = [1:max(max(label))];