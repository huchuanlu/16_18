function [newLabelCR, newLabelSVM, newLabelSUM, newLabelMUL] = classification_cr_svm(data, label, model)

winH = model.winH;
winW = model.winW;
%
[height,width,bands] = size(data);
indexMatrix = [1:height*width]';
indexMatrix = reshape(indexMatrix, [height, width]);
indexMatrix(label==0) = 0;
indexMatrix = blkdiag(zeros(winH,winW), indexMatrix, zeros(winH,winW));
%
dataMatrix = reshape(data, height*width, bands)';
dataMatrix = dataMatrix./(repmat(sqrt(sum(dataMatrix.^2)), [bands 1])+eps);
%
newLabel = zeros(height, width);
newLabelCR  = zeros(height, width);
newLabelSVM = zeros(height, width);
newLabelSUM = zeros(height, width);
newLabelMUL = zeros(height, width);
%
for ii = 1:height
%     [num2str(ii) '/' num2str(height)]
    for jj = 1:width
        neighborIndex = indexMatrix(ii:ii+2*winH, jj:jj+2*winW);
        neighborIndex = neighborIndex(:);
        neighborIndex(neighborIndex==0) = [];
        dataNeighbor = dataMatrix(:,neighborIndex);
        if  isempty(dataNeighbor)
            newLabel(ii,jj) = 0;
            continue;
        end
        nLabels = max(max(model.label));
        residue = zeros(nLabels,1);
        coeff = model.projection*dataNeighbor;
        for num = 1:nLabels
            labelIndex = (model.label==num);
            labelIndex = labelIndex(:);
            residue(num) = norm(dataNeighbor-model.dictionary(:, labelIndex)*coeff(labelIndex,:), 'fro')^2;
        end
        crpro = exp(-(residue-mean(residue))/std(residue));
        crpro   = crpro./sum(crpro);
        %
        coeff = sum(coeff,2);
        coeff = coeff./(norm(coeff)+eps);
        [~, ~, svmpro] = svmpredict(0, coeff', model.svmmodel, ['-b 1 -q']);
        %
        [~, newLabelCR(ii,jj)]  = max(crpro);
        [~, newLabelSVM(ii,jj)] = max(svmpro);
        prosum = 0.5*crpro + 0.5*svmpro';
        [~, newLabelSUM(ii,jj)] = max(prosum);
        promul = crpro.*svmpro';
        [~, newLabelMUL(ii,jj)] = max(promul);
    end
end
newLabelCR(label==0)  = 0;
newLabelSVM(label==0) = 0;
newLabelSUM(label==0) = 0;
newLabelMUL(label==0) = 0;