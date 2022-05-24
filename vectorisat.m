clear;clc;
frames = 10;
numLagsPoints = 30;
numHorizon = 30;
trainVehLen = 0;
predData = readtable(fullfile('C:\PhD_Files\M_Exam','predictedOutput_Unregulated_wULI_MCI_PadV_3-3_[256, 256]-0.0001-32.csv'));
testData = readtable(fullfile('C:\PhD_Files\M_Exam','testingData_Y_unregulated_wULI_MCI_PadV_3-3_junctionCheck.csv'));
testData = testData{:,1:2};
tic;
predData = predData{2:end, :};
dataLen = size(predData, 1);
predData = reshape(predData, numHorizon, 2, int32(dataLen/numHorizon));
testData = reshape(testData, numHorizon, 2, int32(dataLen/numHorizon));
%%
euclDist = zeros(3, 1);
idx = 1;
for i=1:10:30  
    a = predData(i:idx*10, :, :);
    a = reshape(a, size(a, 3)*10, 2);
    b = testData(i:idx*10, :, :);
    b = reshape(b, size(b, 3)*10, 2);
    euclDist(idx) = mean(sqrt(sum((a(:, 1:2)-b(:, 1:2)).^2,2)));
    idx = idx+1;
end
toc;
%% un parallel
tic;
predData = predData{2:end, :};
dataLen = size(predData, 1);
predData = reshape(predData, numHorizon, 2, int32(dataLen/numHorizon));
testData = reshape(testData, numHorizon, 2, int32(dataLen/numHorizon));
euclDist = zeros(10960630, 3);
idx = 1;
for i=1:10:30  
    a = predData(i:idx*10, :, :);
    a = reshape(a, size(a, 3)*10, 2);
    b = testData(i:idx*10, :, :);
    b = reshape(b, size(b, 3)*10, 2);
    for j=1:size(a, 1)
        euclDist(j, idx) = pdist2(a(j, :),b(j, :));
    end
    idx = idx+1;
end
toc;