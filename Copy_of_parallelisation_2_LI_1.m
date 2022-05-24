% p=parpool('local', 4);
% clear;clc;
p=parpool('local', 4);
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
%%
tic;
workers = 4;
G = findgroups(trainingVehiclesData.veh_id);
trainingVehiclesData(:, "grpID") = table(G);
[gnums, sgnums] = sort(G);
groupID = [gnums, sgnums];
[workerID, endIdx]= discretize(unique(G), linspace(min(G), max(G), workers + 1)); 
groupedData =  cell(1, workers);
processeddata = cell(1, workers);
uniqueGroups = unique(G);
trainVehIndexes = cell(size(uniqueGroups, 1), 1);
parfor i=1:size(uniqueGroups, 1)
    trainVehIndexes{i} = {uniqueGroups(i) groupID((groupID(:, 1)==uniqueGroups(i)), 2)};
end
trainVehIndexes = cat(1, trainVehIndexes{:});
trainVehIndexes = cell2table(trainVehIndexes);
for i=1:workers
    focusIndexes = trainVehIndexes{(trainVehIndexes.trainVehIndexes1>=endIdx(i)) & (trainVehIndexes.trainVehIndexes1<endIdx(i+1)), 2};
    focusIndexes = cat(1, focusIndexes{:});
    groupedData(i) = {trainingVehiclesData(focusIndexes, ["time", "x", "y", "speed", 'acc', "angle", ...
            "leadVehicleSpeed", "leadVehicleX", "leadVehicleY"])}; 

end
toc
%%
tic
ticBytes(gcp);
parfor i=1:workers
    data = groupedData{i};    
    processeddata{i} = prepareData(data, meanData, stdData);
%     training_X{i} = X;
%     training_Y{i} = y;
end
tocBytes(gcp);
% trainData_X = cat(1, training_X{:});
% trainData_Y = cat(1, training_Y{:});
toc;
memValue = memory;
endMem = memValue.MemUsedMATLAB;
%%

function output = prepareData(data, meanData, stdData)
    output = cell(size(data, 1), 1);
    for k=1:size(data, 1)
        curData = data{k};
        curData = sortrows(curData, "time");
        curData = removevars(curData,"time");
        curData = (curData{:, :}-meanData{:, :})./stdData{:, :};
        % optimised
        rollingWindow = 30;
        dataLen = size(curData, 1)-30;
        rollingWinIDX = dataLen - rollingWindow + 1;
        output{k} = [curData(hankel(1:rollingWinIDX, rollingWinIDX:dataLen), :) curData(hankel(1:rollingWinIDX, rollingWinIDX:dataLen)+30, 1:2)];
    end
end
