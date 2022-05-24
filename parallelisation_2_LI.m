% p=parpool('local', 4);
% clear;clc;
% p=parpool('local', 4);
memValue = memory;
initialMem = memValue.MemUsedMATLAB;
frames = 10;
numLagsPoints = 30;
numHorizon = 30;
trainVehLen = 0;
fullData = readtable(fullfile('C:\PhD_Files\M_Exam','vehicles_data.csv'));
standardiseData = readtable(fullfile('C:\PhD_Files\M_Exam','standardise.csv'));
meanData = (standardiseData(1, 2:size(standardiseData, 2)));
stdData = (standardiseData(2, 2:size(standardiseData, 2)));
uniqueVehicles = unique(fullData.veh_id);
trainSplit = int32(size(uniqueVehicles, 1) * 0.65);
trainingVehicles = uniqueVehicles(1:trainSplit);        
training_X = cell(size(trainingVehicles, 1), 1);
training_Y = cell(size(trainingVehicles, 1), 1);
trainingVehiclesData = fullData(ismember(fullData.veh_id, trainingVehicles),:);
tic;
G = findgroups(trainingVehiclesData.veh_id);
[gnums, sgnums] = sort(G);
uniqueGroups = unique(G);
groupID = [gnums, sgnums];
groupedData =  cell(size(trainingVehicles, 1), 2);
for i=1:size(uniqueGroups, 1)    
    data = trainingVehiclesData(groupID(groupID(:, 1)==uniqueGroups(i), 2), :);
    data = sortrows(data, "time");
    data = data(:,["x", "y", "speed", 'acc', "angle", ...
        "leadVehicleSpeed", "leadVehicleX", "leadVehicleY"]);
    [X, y] = prepareData(data, meanData, stdData);
    training_X{i} = X;
    training_Y{i} = y;
end
trainData_X = cat(1, training_X{:});
trainData_Y = cat(1, training_Y{:});
toc;
memValue = memory;
endMem = memValue.MemUsedMATLAB;
%%

function [X, y] = prepareData(data, meanData, stdData)   
    data = (data{:, :}-meanData{:, :})./stdData{:, :};
    % optimised
    rollingWindow = 30;
    dataLen = size(data, 1)-30;
    rollingWinIDX = dataLen - rollingWindow + 1;
    X = data(hankel(1:rollingWinIDX, rollingWinIDX:dataLen), :);
    y = data(hankel(1:rollingWinIDX, rollingWinIDX:dataLen)+30, 1:2);
end
