clear;clc;
frames = 10;
numLagsPoints = 30;
numHorizon = 30;
trainVehLen = 0;
fullData = readtable(fullfile('C:\PhD_Files\M_Exam','vehicles_data.csv'));
standardiseData = readtable(fullfile('C:\PhD_Files\M_Exam','standardise.csv'));
meanData = table2array(standardiseData(1, 2:size(standardiseData, 2)));
stdData = table2array(standardiseData(2, 2:size(standardiseData, 2)));
uniqueVehicles = unique(fullData.veh_id);
trainSplit = int32(size(uniqueVehicles, 1) * 0.65);
trainingVehicles = uniqueVehicles(1:trainSplit);        
training_X = cell(size(trainingVehicles, 1), 1);
training_Y = cell(size(trainingVehicles, 1), 1);

trainingVehiclesData = fullData(ismember(fullData.veh_id, trainingVehicles),:);
%%
workers = 4;
% tic;
G = findgroups(trainingVehiclesData.veh_id);
[gnums, sgnums] = sort( G );
groupID = [gnums, sgnums];
[workerID, endIdx]= discretize(unique(G), linspace(min(G), max(G), workers + 1)); 
processeddata = cell(1, workers);
tic
ticBytes(gcp);
parfor i = 1:workers
    focusIndexes = groupID((groupID(:,1)>=endIdx(i)) &(groupID(:,1)<endIdx(i+1)), :);
    processeddata{i} = prepareData(trainingVehiclesData, focusIndexes);
end
tocBytes(gcp)
toc
%%
function outputData = prepareData(data, focusIndexes)         
    curVehicles = unique(focusIndexes(:, 1));
    rollingWindow = 30;
    outputData = cell(size(curVehicles, 1), 2);
%     for k=1:size(curVehicles, 1)
    curdata = data(focusIndexes(focusIndexes(:, 1)=='carIn29589:1', 2), :);        
    dataLen = size(curdata, 1)-30;
    rollingWinIDX = dataLen - rollingWindow + 1;
    outputData{k} = {curdata(hankel(1:rollingWinIDX, rollingWinIDX:dataLen), :), curdata(hankel(1:rollingWinIDX, rollingWinIDX:dataLen)+30, 1:2)};
%     end    
end
