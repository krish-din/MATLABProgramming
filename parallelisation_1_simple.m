memValue = memory;
initialMem = memValue.MemUsedMATLAB;
frames = 10;
numLagsPoints = 30;
numHorizon = 30;
trainVehLen = 0;
fullData = readtable(fullfile('C:\PhD_Files\M_Exam','vehicles_data.csv'));
stdData = readtable(fullfile('C:\PhD_Files\M_Exam','standardise.csv'));
uniqueVehicles = unique(fullData.veh_id);
trainSplit = int32(size(uniqueVehicles, 1) * 0.65);
trainingVehicles = uniqueVehicles(1:trainSplit);        
% training_X = cell(size(trainingVehicles, 1), 1);
% training_Y = cell(size(trainingVehicles, 1), 1);
trainingVehiclesData = fullData(ismember(fullData.veh_id, trainingVehicles),:);
tic;
% parpool("threads");
for i=1:size(trainingVehicles, 1)
    data = trainingVehiclesData(trainingVehiclesData.veh_id==string(trainingVehicles{i}), :);
    data = sortrows(data, "time");
    data = data(:,["x", "y", "speed", 'acc', "angle", ...
        "leadVehicleSpeed", "leadVehicleX", "leadVehicleY"]);
    [X, y] = prepareData(data, stdData);
    training_X{i} = X;
    training_Y{i} = y;
end
trainData_X = cat(1, training_X{:});
trainData_Y = cat(1, training_Y{:});
toc;
memValue = memory;
endMem = memValue.MemUsedMATLAB;
%%
function [X, y] = prepareData(data, stdData)        
    data = (table2array(data) - table2array(stdData(1, 2:size(stdData, 2))))./table2array(stdData(2, 2:size(stdData, 2)));
%     totalLen = size(data, 1) - numLagsPoints - numHorizon + 1;
%     X = zeros(totalLen*30, 8);
%     y = zeros(totalLen*30, 2);    
%     startIDX = 1;    
%     for i= 1: int32(totalLen)
%         endIDX = startIDX+29;
%         X(startIDX:endIDX, :) = data(i:numLagsPoints + i-1, :);
%         y(startIDX:endIDX, :) = data(numLagsPoints + i:numLagsPoints + i + numHorizon - 1, 1:2);
%         startIDX = endIDX+1;
%     end
    % optimised
    rollingWindow = 30;
    dataLen = size(data, 1)-30;
    rollingWinIDX = dataLen - rollingWindow + 1;
    X = data(hankel(1:rollingWinIDX, rollingWinIDX:dataLen), :);
    y = data(hankel(1:rollingWinIDX, rollingWinIDX:dataLen)+30, 1:2);
end
