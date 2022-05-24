memValue = memory;
initialMem = memValue.MemUsedMATLAB;
frames = 10;
numLagsPoints = 30;
numHorizon = 30;
trainVehLen = 0;
df = readtable(fullfile('C:\PhD_Files\M_Exam','vehicles_data.csv'));
df = df(:, ["time", "x", "y", "speed", 'acc', "angle", ...
        "leadVehicleSpeed", "leadVehicleX", "leadVehicleY", "veh_id"]);
standardiseData = readtable(fullfile('C:\PhD_Files\M_Exam','standardise.csv'));
meanData = table2array(standardiseData(1, 2:size(standardiseData, 2)));
stdData = table2array(standardiseData(2, 2:size(standardiseData, 2)));
uniqueVehicles = unique(df.veh_id);
trainSplit = int32(size(uniqueVehicles, 1) * 0.65);
trainingVehicles = uniqueVehicles(1:trainSplit);        
training_X = cell(size(trainingVehicles, 1), 1);
training_Y = cell(size(trainingVehicles, 1), 1);
trainingVehData = df(ismember(df.veh_id,trainingVehicles),:);
%%
tic;
% finding groups
G = findgroups(trainingVehData.veh_id);
% doing split apply combine routine for UDF
outPut = splitapply(@prepareData, ...
    trainingVehData(:, ["time", "x", "y", "speed", 'acc', "angle", ...
        "leadVehicleSpeed", "leadVehicleX", "leadVehicleY"]), G);
trainData_X = outPut(:, 1);
trainData_X = cat(1, trainData_X{:});
trainData_Y = outPut(:, 2);
trainData_Y = cat(1, trainData_Y{:});
trainData_X = (trainData_X - meanData)./stdData;
trainData_Y = (trainData_Y - meanData(1:2))./stdData(1:2);
toc;
memValue = memory;
endMem = memValue.MemUsedMATLAB;
%%
function out = prepareData(data1, data2, data3, data4, data5, data6, data7, data8, data9)    
    data = [data1, data2, data3, data4, data5, data6, data7, data8, data9];        
    data = sortrows(data, 1);
    data(:, 1) = [];
    rollingWindow = 30;
    dataLen = size(data, 1)-30;
    rollingWinIDX = dataLen - rollingWindow + 1;
    X = data(hankel(1:rollingWinIDX, rollingWinIDX:dataLen), :);
    y = data(hankel(1:rollingWinIDX, rollingWinIDX:dataLen)+30, 1:2);
    out = {X,y};
end