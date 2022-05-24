import pandas as pd
import numpy as np
import math
np.random.seed(0)
frames = 10
newFrames = 10
numLagsPoints = 30
numHorizon = 30
totalHorizon = 30
dfSizeReduction = math.floor(frames / newFrames)
framePerSecond = frames // dfSizeReduction
df = pd.read_csv(r"C:\PhD_Files\M_Exam\vehicles_data.csv")
stdData = pd.read_csv(r'C:\PhD_Files\M_Exam\standardise.csv')
uniqueVehicles = df.veh_id.unique()
trainSplit = int(len(uniqueVehicles) * 0.65)
trainingVehicles = uniqueVehicles[:trainSplit+1]
training_X = []
training_Y = []
def prepareData(data, n_lags, n_seq, initial):
    X = []
    y = []
    numHorizon = round(n_seq * framePerSecond)
    numLagsPoints = round(n_lags * framePerSecond)

    totalLen = data.shape[0] - numLagsPoints - numHorizon + 1
    # data = np.divide(np.subtract(data, np.asarray(allDataMin)), np.asarray(allDataMax - allDataMin))
    data = np.divide(np.subtract(data, np.asarray(allDataMean)), np.asarray(allDataStd))
    for i in range(int(totalLen)):
        X.append(data.iloc[i:numLagsPoints + i].values)
        y.append(data.loc[numLagsPoints + i:numLagsPoints + i + numHorizon - 1,
                 ["x", "y"]].values)
    return X, y

allDataMean = stdData.iloc[0, 1:]
allDataStd = stdData.iloc[1, 1:]

import time
startTime = time.time()
df_Filtered = df[df.veh_id.isin(trainingVehicles)]
groupData = df_Filtered.groupby("veh_id")
for vehID, curDF in groupData:
    curDF = curDF.loc[:, ["time", "x", "y", "speed", 'acc', "angle",
                            "leadVehicleSpeed", "leadVehicleX", "leadVehicleY"]
                            ]
    curDF.sort_values("time", inplace=True)
    curDF.reset_index(drop=True, inplace=True)
    curDF.drop("time", axis=1, inplace=True)
    x, y = prepareData(curDF, (numLagsPoints / framePerSecond), (numHorizon / framePerSecond), False)

    training_X.append(np.concatenate(x))
    training_Y.append(np.concatenate(y))    
endTime = time.time()
#%%

#%%
import numpy as np
import pandas as pd
frames = 10
newFrames = 10
numLagsPoints = 30
numHorizon = 30
totalHorizon = 30
framePerSecond = 10
def euclDistCalc(actual, pred, actuaCol=["act_x", "act_y"], predCol=["pred_x", "pred_y"]):
    return np.linalg.norm(actual[actuaCol].values - pred[predCol].values,
                          axis=1)
eucledianDist = {}
eucledianDist1 = {}
df = pd.read_csv(r"C:\PhD_Files\projects\TrajectoryPrediction-CAvoid\SUMO_Related\collisionAvoidance\dataFiles\predictedOutput_Unregulated_wULI_MCI_PadV_3-3_[256, 256]-0.0001-32.csv")
testdf = pd.read_csv(r"C:\PhD_Files\projects\TrajectoryPrediction-CAvoid\SUMO_Related\collisionAvoidance\dataFiles\testingData_Y_unregulated_wULI_MCI_PadV_3-3_junctionCheck.csv")
testdf = testdf.iloc[:, :2]
testdf = np.array(testdf).reshape(int(len(testdf) / numLagsPoints), 30, 2)
df = np.array(df).reshape(int(len(df) / numLagsPoints), 30, 2)
actualOutput = [{i + 1: pd.DataFrame([]) for i in range(int(numHorizon / framePerSecond))}]
predOutput = [{i + 1: pd.DataFrame([]) for i in range(int(numHorizon / framePerSecond))}]
import time
startTime = time.time()
for j in range(int(numHorizon / framePerSecond)):
    predData = df[:, j * framePerSecond:(j + 1) * framePerSecond, :]
    predDataDF = pd.DataFrame(predData.reshape(int(len(predData) * framePerSecond), 2))
    predOutput[0][j + 1] = predDataDF
    predOutput[0][j + 1] = predOutput[0][j + 1].round(2)
    actualData = testdf[:, j * framePerSecond:(j + 1) * framePerSecond, :]
    actualDataDF = pd.DataFrame(actualData.reshape(int(len(actualData) * framePerSecond), 2))
    actualOutput[0][j + 1] = actualDataDF
    actualOutput[0][j + 1] = actualOutput[0][j + 1].round(2)
del predData, predDataDF, actualDataDF, actualData

for j in range(int(numHorizon / framePerSecond)):
    # eucledianDist[j + 1] = {}
    eucledianDist1[j + 1] = {}
    # a = actualOutput[0][j + 1].copy()
    # b = predOutput[0][j + 1].copy()
    # c = pd.concat([a, b], axis=1)
    # c.columns = ["act_x", "act_y", "pred_x", "pred_y"]
    # del a, b
    # c["dist"] = c.apply(lambda l: np.linalg.norm(np.array([l[0], l[1]]) - np.array([l[2], l[3]])), axis=1)
    # eucledianDist[j + 1]["dist"] = c["dist"].mean()
    a = actualOutput[0][j + 1].copy()
    a.columns = ["act_x", "act_y"]
    b = predOutput[0][j + 1].copy()
    b.columns = ["pred_x", "pred_y"]
    c = euclDistCalc(a, b)
    eucledianDist1[j + 1]["dist"] = c.mean()
    del a, b

endTime = time.time()
