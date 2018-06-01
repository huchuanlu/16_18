function [results] = cr_svm(dataBase, num, winsz)

addpath(genpath('./BasisFunctions'));
load(['./Data/' dataBase '_' num2str(num) '.mat']);
%
options = [];
options.param = [];
options.param.K = num;
options.winH = winsz;
options.winW = winsz;
[model] = train_model(data, label, trainIndex, options);
%
model.winH  = winsz;
model.winW  = winsz;
[newLabelCR, newLabelSVM, newLabelSUM, newLabelMUL] = classification_cr_svm(data, label, model);
%
nLabels = model.nLabels;
[oAccuracyCR,  kAccuracyCR,  aAccuracyCR,  class_accuracyCR]  = calcError(label(testIndex), newLabelCR(testIndex),  nLabels);
[oAccuracySVM, kAccuracySVM, aAccuracySVM, class_accuracySVM] = calcError(label(testIndex), newLabelSVM(testIndex), nLabels);
[oAccuracySUM, kAccuracySUM, aAccuracySUM, class_accuracySUM] = calcError(label(testIndex), newLabelSUM(testIndex), nLabels);
[oAccuracyMUL, kAccuracyMUL, aAccuracyMUL, class_accuracyMUL] = calcError(label(testIndex), newLabelMUL(testIndex), nLabels);
%
results = [];
%
results.oAccuracyCR = oAccuracyCR;
results.kAccuracyCR = kAccuracyCR;
results.aAccuracyCR = aAccuracyCR;
results.class_accuracyCR = class_accuracyCR;
results.newLabelCR  = newLabelCR;
%
results.oAccuracySVM = oAccuracySVM;
results.kAccuracySVM = kAccuracySVM;
results.aAccuracySVM = aAccuracySVM;
results.class_accuracySVM = class_accuracySVM;
results.newLabelSVM  = newLabelSVM;
%
results.oAccuracySUM = oAccuracySUM;
results.kAccuracySUM = kAccuracySUM;
results.aAccuracySUM = aAccuracySUM;
results.class_accuracySUM = class_accuracySUM;
results.newLabelSUM  = newLabelSUM;
%
results.oAccuracyMUL = oAccuracyMUL;
results.kAccuracyMUL = kAccuracyMUL;
results.aAccuracyMUL = aAccuracyMUL;
results.class_accuracyMUL = class_accuracyMUL;
results.newLabelMUL  = newLabelMUL;