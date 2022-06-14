import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import xmltodict
import mujoco_py as mujoco

import itertools as it
from collections import OrderedDict
import numpy as np
from env.multiAgentMujocoEnv import TransitionFunctionWithoutXPosForExp, RewardSheep, RewardWolf, Observe, IsCollision, getPosFromAgentState, \
    getVelFromAgentState,PunishForOutOfBound,ReshapeAction, TransitionFunctionWithoutXPos, ResetUniformWithoutXPosForLeashed, PunishForOutOfBoundVarRange

from src.maddpg.trainer.myMADDPG import ActOneStep, BuildMADDPGModels, actByPolicyTrainNoisy

from src.functionTools.loadSaveModel import saveToPickle, restoreVariables,GetSavePath
from src.functionTools.trajectory import SampleExpTrajectory,SampleExpTrajectoryWithAllFrames
from src.functionTools.editEnvXml import transferNumberListToStr,MakePropertyList,changeJointProperty
from src.visualize.visualizeMultiAgent import Render



wolfColor = np.array([0.85, 0.35, 0.35])
sheepColor = np.array([0.35, 0.85, 0.35])
masterColor= np.array([0.35, 0.35, 0.85])
distractorColor = np.array([0.35, 0.85, 0.85])
blockColor = np.array([0.25, 0.25, 0.25])




def generateSingleCondition(condition):
    debug = 0
    if debug:


        damping=2.0
        frictionloss=0.0
        masterForce=1.0

        numWolves = 1
        numSheeps = 1
        numMasters = 1
        numDistractor = 1
        maxTimeStep = 25

        maxEpisode = 60000
        saveTraj=True
        saveImage=True
        visualizeMujoco=False
        visualizeTraj = True
        makeVideo=True
    else:

        # print(sys.argv)
        # condition = json.loads(sys.argv[1])
        damping = 0.5
        frictionloss = 1.4
        offset = 0.0
        killZoneRatio = 4.0
        distractKillZoneRatio = 0.0
        ropePunishWeight = 0.3
        ropeLength = float(condition['ropeLength'])
        masterMass = 1.0
        masterForce = float(condition['masterForce'])
        masterPunishRange = float(condition['masterPunishRange'])
        masterPullDistance = float(condition['forceAllowedDistance'])
        masterPullPunish = 1.0
        masterPullForce = float(condition['masterPullForce'])
        sheepPunishRange = float(condition['sheepPunishRange'])
        sheepForce = float(condition['sheepForce'])
        masterPullDistanceForSheep = float(condition['forceAllowedDistanceForSheep'])
        wolfForce = float(condition['wolfForce'])
        distractorNoise = float(condition['distractorNoise'])
        dt = 0.02
        offsetFrame = int (offset/dt)

        maxEpisode = 90000
        evaluateEpisode = 90000
        numWolves = 1
        numSheeps = 1
        numMasters = 1
        numDistractor = 2
        maxTimeStep = 25

        noiseDistractor=False
        if noiseDistractor:
            distractorNoise = 32.0

        saveTraj=True
        saveImage=True
        visualizeMujoco=True
        visualizeTraj = True
        makeVideo=True

    evalNum = 4
    maxRunningStepsToSample = 100
    modelSaveName = 'MasterForDistractorWolveSpeedUpAndSheepDistancewithFrictionChanged'
    # modelSaveName = 'expTrajMADDPGMujocoEnvWithRopeAdd2Distractors'
    print("maddpg: , saveTraj: {}, visualize: {},damping; {},frictionloss: {}".format( str(saveTraj), str(visualizeMujoco),damping,frictionloss))
    wolvesID = [0]
    sheepsID = [1]
    masterID = [2]
    distractorID = [3,4]
    hideIdList = [3]# distractorID + sheepsID
    numAgent=5


    numAgent = numWolves + numSheeps + numMasters +  numDistractor
    numAgentForDarw =  4

    wolvesID = [0]
    sheepsID = [1]
    masterID = [2]
    distractorID = [3,4]

    wolfSize = 0.05
    sheepSize = 0.05
    masterSize = 0.05
    distractorSize = 0.05
    entitiesSizeList = [wolfSize] * numWolves + [sheepSize] * numSheeps + [masterSize] * numMasters + [distractorSize] * numDistractor
    entitiesSizeListForDarw = [wolfSize] * numWolves + [sheepSize] * numSheeps + [masterSize] * numMasters + [distractorSize] * 1

    entitiesMovableList = [True] * numAgent + [False] * numMasters

    killZone = wolfSize * killZoneRatio
    isCollision = IsCollision(getPosFromAgentState, killZone)
    sheepPunishForOutOfBound = PunishForOutOfBoundVarRange(sheepPunishRange)
    punishForOutOfBound = PunishForOutOfBound()
    rewardSheep = RewardSheep(wolvesID, sheepsID, entitiesSizeList, getPosFromAgentState, isCollision, sheepPunishForOutOfBound)
    rewardWolf = RewardWolf(wolvesID, sheepsID, entitiesSizeList, isCollision)
    rewardDistractor = RewardSheep(wolvesID+sheepsID+masterID, distractorID, entitiesSizeList, getPosFromAgentState, isCollision,punishForOutOfBound)
    rewardMaster= lambda state, action, nextState: [-reward  for reward in rewardWolf(state, action, nextState)]
    rewardFunc = lambda state, action, nextState: \
        list(rewardWolf(state, action, nextState)) + list(rewardSheep(state, action, nextState))\
        + list(rewardMaster(state, action, nextState) )+ list(rewardDistractor(state, action, nextState))

    physicsDynamicsPath=os.path.join(dirName,'..','..','env','xml','leased2Distractor_masterMass={}_ropeLength={}.xml'.format(masterMass,ropeLength))
    print('loadEnv:{}'.format(physicsDynamicsPath))
    with open(physicsDynamicsPath) as f:
        xml_string = f.read()

    makePropertyList=MakePropertyList(transferNumberListToStr)

    geomIds=[1,2,3,4,5]
    keyNameList=[0,1]
    valueList=[[damping,damping]]*len(geomIds)
    dampngParameter=makePropertyList(geomIds,keyNameList,valueList)

    changeJointDampingProperty=lambda envDict,geomPropertyDict:changeJointProperty(envDict,geomPropertyDict,'@damping')

    geomIds=[1,2,3,4,5]
    keyNameList=[0,1]
    valueList=[[frictionloss,frictionloss]]*len(geomIds)
    frictionlossParameter=makePropertyList(geomIds,keyNameList,valueList)
    changeJointFrictionlossProperty=lambda envDict,geomPropertyDict:changeJointProperty(envDict,geomPropertyDict,'@frictionloss')



    envXmlDict = xmltodict.parse(xml_string.strip())
    envXmlPropertyDictList=[dampngParameter,frictionlossParameter]
    changeEnvXmlPropertFuntionyList=[changeJointDampingProperty,changeJointFrictionlossProperty]
    for propertyDict,changeXmlProperty in zip(envXmlPropertyDictList,changeEnvXmlPropertFuntionyList):
        envXmlDict=changeXmlProperty(envXmlDict,propertyDict)

    envXml=xmltodict.unparse(envXmlDict)
    physicsModel = mujoco.load_model_from_xml(envXml)
    physicsSimulation = mujoco.MjSim(physicsModel)

    numKnots = 9
    numAxis = (numKnots + numAgent) * 2
    qPosInit = (0, ) * numAxis
    qVelInit = (0, ) * numAxis
    qPosInitNoise = 0.4
    qVelInitNoise = 0
    tiedAgentId = [0, 2]
    ropePartIndex = list(range(numAgent, numAgent + numKnots))
    maxRopePartLength = 0.06
    reset = ResetUniformWithoutXPosForLeashed(physicsSimulation, qPosInit, qVelInit, numAgent, tiedAgentId,ropePartIndex, maxRopePartLength, qPosInitNoise, qVelInitNoise)
    numSimulationFrames=10
    isTerminal= lambda state: False
    distractorReshapeAction=ReshapeAction(5)
    noiseMean = (0, 0)
    noiseCov = [[distractorNoise, 0], [0, distractorNoise]]
    # x = np.random.multivariate_normal(noiseMean, noiseCov, (1, 1), 'raise')[0]
    class LimitForceMagnitude():
        def __init__(self,entityMaxForceMagnitude=None):
            self.entityMaxForceMagnitude = entityMaxForceMagnitude

        def __call__(self,entityNextForce):
            # print(entityNextForce)
            if self.entityMaxForceMagnitude is not None:
                forceMagnitude = np.sqrt(np.square(entityNextForce[0]) + np.square(entityNextForce[1])) #
            if forceMagnitude > self.entityMaxForceMagnitude:
                entityNextForce = entityNextForce / forceMagnitude * self.entityMaxForceMagnitude

            return np.array(entityNextForce)

    limitForceMagnitude = LimitForceMagnitude(5)

    noiseDistractorAction= lambda state:limitForceMagnitude((distractorReshapeAction(state)+np.random.multivariate_normal(noiseMean, noiseCov, (1, 1), 'raise')[0])[0])
    if noiseDistractor:
          reshapeActionList = [ReshapeAction(wolfForce),ReshapeAction(sheepForce),ReshapeAction(masterForce),noiseDistractorAction,noiseDistractorAction]
    else:
        reshapeActionList = [ReshapeAction(wolfForce),ReshapeAction(sheepForce),ReshapeAction(masterForce),ReshapeAction(15),ReshapeAction(15)]



    transit=TransitionFunctionWithoutXPosForExp(physicsSimulation, numSimulationFrames, visualizeMujoco,isTerminal, reshapeActionList)


    sampleTrajectory = SampleExpTrajectoryWithAllFrames(maxRunningStepsToSample, transit, isTerminal, rewardFunc, reset)


    observeOneAgent = lambda agentID: Observe(agentID, wolvesID, sheepsID + masterID +distractorID, [], getPosFromAgentState, getVelFromAgentState)
    observe = lambda state: [observeOneAgent(agentID)(state) for agentID in range(numAgent)]
    print(reset())

    initObsForParams = observe(reset())
    obsShape = [initObsForParams[obsID].shape[0] for obsID in range(len(initObsForParams))]
    print('24e',obsShape)
    worldDim = 2
    actionDim = worldDim * 2 + 1

    layerWidth = [128, 128]

    # ------------ model ------------------------
    buildMADDPGModels = BuildMADDPGModels(actionDim, numAgent, obsShape)
    modelsList = [buildMADDPGModels(layerWidth, agentID) for agentID in range(numAgent)]

    dataFolder = os.path.join(dirName, '..','..', 'data')
    mainModelFolder = os.path.join(dataFolder,'modelplus')
    # modelFolder = os.path.join(mainModelFolder, modelSaveName,'damping={}_frictionloss={}_killZoneRatio{}_masterForce={}_masterMass={}_ropeLength={}_ropePunishWeight={}'.format(damping,frictionloss,killZoneRatio,masterForce,masterMass,ropeLength,ropePunishWeight))
    
    modelFolder = os.path.join(mainModelFolder, modelSaveName,'sheepForce={}_wolfForce={}_masterforce={}_masterPullForce={}_masterPullDistanceForSheep={}'.format(sheepForce,wolfForce,masterForce,masterPullForce,masterPullDistanceForSheep))

    fileName = "maddpg{}episodes{}step_agent".format(maxEpisode, maxTimeStep)
    
    modelPaths = [os.path.join(modelFolder,  fileName + str(i) +str(evaluateEpisode)+'eps') for i in range(numAgent)]

    [restoreVariables(model, path) for model, path in zip(modelsList, modelPaths)]

    actOneStepOneModel = ActOneStep(actByPolicyTrainNoisy)
    policy = lambda allAgentsStates: [actOneStepOneModel(model, observe(allAgentsStates)) for model in modelsList]
    # offsetFrameList=[0] + [offsetFrame]*3
    offsetFrameList=[offsetFrame,offsetFrame,0,offsetFrame] #wolf sheep master
    for hideId in hideIdList:
        agentList = list(range(numAgent))
        del(agentList[hideId])
        trajList = []
        expTrajList = []
        newTrajList = []
        for _ in range(evalNum):
            # np.random.seed(i)
            traj, expTraj = sampleTrajectory(policy)
            trajList.append(list(traj))
            expTrajList.append((list(expTraj)))
        for i,traj in enumerate(expTrajList):
            newTraj = [[state[agentId] for agentId in agentList]  for state in traj]
            if offsetFrame < 0:
                offsetTraj =  [[newTraj[index+offsetF][i] for i,offsetF in enumerate(offsetFrameList)]  for index in range(-offsetFrame,len(newTraj))]
            else:
                offsetTraj =  [[newTraj[index+offsetF][i] for i,offsetF in enumerate(offsetFrameList)]  for index in range(len(newTraj)-offsetFrame)]
            newTrajList.append(offsetTraj)

        print('save',newTrajList[0][0])
        # saveTraj
        if saveTraj:
            # trajFileName = "maddpg{}wolves{}sheep{}blocks{}eps{}step{}Traj".format(numWolves, numSheeps, numMasters, maxEpisode, maxTimeStep)

            trajectoriesSaveDirectory= os.path.join(dataFolder,'trajectory',modelSaveName,'ENDnoiseOffsetWithMasterPull')
            if not os.path.exists(trajectoriesSaveDirectory):
                os.makedirs(trajectoriesSaveDirectory)

            trajectorySaveExtension = '.pickle'
            fixedParameters = {'evalNum':evalNum,'evaluateEpisode':evaluateEpisode,'hideId':hideId}
            generateTrajectorySavePath = GetSavePath(trajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
            trajectorySavePath = generateTrajectorySavePath(condition)
            saveToPickle(trajList, trajectorySavePath)

            expTrajectoriesSaveDirectory = os.path.join(dataFolder, 'Exptrajectory', modelSaveName,'ENDnoiseOffsetWithMasterPull')
            if not os.path.exists(expTrajectoriesSaveDirectory):
                os.makedirs(expTrajectoriesSaveDirectory)

            generateExpTrajectorySavePath = GetSavePath(expTrajectoriesSaveDirectory, trajectorySaveExtension, fixedParameters)
            expTrajectorySavePath = generateExpTrajectorySavePath(condition)
            saveToPickle(newTrajList, expTrajectorySavePath)

        # visualize
        if visualizeTraj:

            # pictureFolder = os.path.join(dataFolder, 'demo', modelSaveName,'normal','damping={}_frictionloss={}_masterForce={}'.format(damping,frictionloss,masterForce))
            pictureFolder = os.path.join(dataFolder, 'demo', modelSaveName,'normal','sheepForce={}_wolfForce={}_masterforce={}_masterPullForce={}_masterPullDistanceForSheep={}'.format(sheepForce,wolfForce,masterForce,masterPullForce,masterPullDistanceForSheep))
            if not os.path.exists(pictureFolder):
                os.makedirs(pictureFolder)
            else:
                return 1
            entitiesColorList = [wolfColor] * numWolves + [sheepColor] * numSheeps + [masterColor] * numMasters + [distractorColor] * numDistractor
            render = Render(entitiesSizeListForDarw, entitiesColorList, numAgentForDarw,pictureFolder,saveImage, getPosFromAgentState)
            trajToRender = np.concatenate(newTrajList)
            print(np.size(trajToRender,0))
            render(trajToRender)


def main():

    manipulatedVariables = OrderedDict()
    # manipulatedVariables['damping'] = [0.5]#[0.0, 1.0]
    # manipulatedVariables['frictionloss'] =[1.0]# [0.0, 0.2, 0.4]
    # manipulatedVariables['masterForce'] = [10.0]
    # manipulatedVariables['offset'] = [-2,-1,-0.5, 0 ,0.5,1,2]
    # manipulatedVariables['distractorNoise']=[3.0]


    manipulatedVariables['ropeLength'] = [0.04] #ssr-1,Xp = 0.06; ssr-3 =0.09
 #ssr-1, ssr-3 = 1.0; Xp = 2.0
    manipulatedVariables['masterPunishRange'] = [0.5]
    manipulatedVariables['forceAllowedDistance'] = [0.3]
    manipulatedVariables['masterForce'] = [17.0,19.0]
    manipulatedVariables['masterPullForce'] = [10.0]
    manipulatedVariables['sheepPunishRange'] = [0.6]
    manipulatedVariables['sheepForce'] = [10.0]
    manipulatedVariables['forceAllowedDistanceForSheep'] = [1.5]
    manipulatedVariables['wolfForce'] = [7.0]
    manipulatedVariables['distractorNoise'] = [20.0]

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    conditions = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]
    for condition in conditions:
        # print(condition)
        generateSingleCondition(condition)
        # try:
            # generateSingleCondition(condition)
        # except:
            # continue

if __name__ == '__main__':
    main()
                                                 
