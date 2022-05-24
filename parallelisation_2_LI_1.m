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
trainingVehiclesData = fullData(ismember(fullData.veh_id, trainingVehicles),:);
%%
tic
workers = 4;
G = findgroups(trainingVehiclesData.veh_id);
trainingVehiclesData(:, "grpID") = table(G);
[workerID, endIdx]= discretize(unique(G), int32(linspace(min(G), max(G)+1, ...
    workers + 1))); 
groupedData =  cell(1, workers);
processeddata = cell(1, workers);
for i=1:workers
    groupedData(i) = {trainingVehiclesData( ...
        (trainingVehiclesData.grpID>=endIdx(i)) & ...
        (trainingVehiclesData.grpID<endIdx(i+1)), ...
        ["time", "x", "y", "speed", 'acc', "angle", ...
            "leadVehicleSpeed", "leadVehicleX", "leadVehicleY", "grpID"])}; 
end
%%
parfor i=1:workers
    data = groupedData{i};    
    processeddata{i} = prepareData(data, meanData, stdData);
end
toc;
tic
processeddata = cat(1, processeddata{:});
processeddata = cat(1, processeddata{:});
trainData_X = processeddata(:, 1:end-2);
trainData_Y = processeddata(:, end-1:end);
toc;
memValue = memory;
endMem = memValue.MemUsedMATLAB;
%%

function output = prepareData(data, meanData, stdData)
    uniqueGroups = unique(data.grpID);
    output = cell(size(uniqueGroups, 1), 1);
    parfor k=1:size(uniqueGroups, 1)
        curData = data(data.grpID==uniqueGroups(k), :);
        curData = sortrows(curData, "time");
        curData = removevars(curData,["time", "grpID"]);
        curData = (curData{:, :}-meanData{:, :})./stdData{:, :};
        rollingWindow = 30;
        dataLen = size(curData, 1)-30;
        rollingWinIDX = dataLen - rollingWindow + 1;
        output{k} = [curData(hankel(1:rollingWinIDX, rollingWinIDX:dataLen), :), curData(hankel(1:rollingWinIDX, rollingWinIDX:dataLen)+30, 1:2)];
    end
end
