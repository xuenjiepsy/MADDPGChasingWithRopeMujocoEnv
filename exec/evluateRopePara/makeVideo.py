
import os
dirName = os.path.dirname(__file__)

import cv2
from collections import OrderedDict
import itertools as it


def makeVideo(condition):
    debug = 0
    if debug:

        damping=2.0
        frictionloss=0.0
        masterForce=1.0

        numTrajToSample=2
        maxRunningStepsToSample=100
    else:
        # print(sys.argv)
        # condition = json.loads(sys.argv[1])
        # damping = float(condition['damping'])
        # frictionloss = float(condition['frictionloss'])
        # masterForce = float(condition['masterForce'])
        # # distractorNoise = float(condition['distractorNoise'])
        # offsetFrame = int(condition['offsetFrame'])
        # damping = float(condition['damping'])
        # frictionloss = float(condition['frictionloss'])
        masterForce = float(condition['masterForce'])
        offset = 0.0
        frictionloss = float(condition['frictionloss'])
        killZoneRatio = float(condition['killZone'])
        ropePunishWeight = float(condition['ropePunishWeight'])
        ropeLength = float(condition['ropeLength'])
        masterMass = float(condition['masterMass'])
        masterPullDistance = float(condition['forceAllowedDistance'])
        masterPullPunish = float(condition['forceForbiddenPunish'])
        masterPullForce = float(condition['masterPullForce'])
        sheepPunishRange = float(condition['sheepPunishRange'])
        masterPunishRange = float(condition['masterPunishRange'])
        sheepForce = float(condition['sheepForce'])
        masterPullDistanceForSheep = float(condition['forceAllowedDistanceForSheep'])
        wolfForce = float(condition['wolfForce'])
        dt = 0.02
        offsetFrame = int (offset/dt)

        # damping = float(condition['damping'])
        # frictionloss = float(condition['frictionloss'])
        masterForce = float(condition['masterForce'])
        # distractorNoise = float(condition['distractorNoise'])
        # offset = float(condition['offset'])
        # hideId = int(condition['hideId'])

        numTrajToSample=4
        maxRunningStepsToSample=1000


    dataFolder = os.path.join(dirName, '..','..', 'data')
    mainDemoFolder = os.path.join(dataFolder,'demo')
    # videoFolder=os.path.join(mainDemoFolder, 'expTrajMADDPGMujocoEnvWithRopeAdd2Distractors','normal')
    videoFolder=os.path.join(mainDemoFolder, 'MasterForDistractorWolveSpeedUpAndSheepDistancewithFrictionChanged','normal')
    # videoFolder=os.path.join(mainDemoFolder, 'expTrajMADDPGMujocoEnvWithRopeAddDistractor_wolfHideSpeed','CrossSheep',)
    # videoFolder=os.path.join(mainDemoFolder, '2expTrajMADDPGMujocoEnvWithRopeAddDistractor_wolfHideSpeed','noise','NoiseDistractor')
    # videoFolder=os.path.join(mainDemoFolder, '2expTrajMADDPGMujocoEnvWithRopeAddDistractor_wolfHideSpeed','OffsetWolfForward')
    # videoFolder=os.path.join(mainDemoFolder, 'demo', 'expTrajMADDPGMujocoEnvWithRopeAdd2DistractorsWithRopePunish','normal',)
    if not os.path.exists(videoFolder):
        os.makedirs(videoFolder)
    # videoPath= os.path.join(videoFolder,'MADDPGMujocoEnvWithRopeAdd2Distractor_damping={}_frictionloss={}_masterForce={}.avi'.format(damping,frictionloss,masterForce))
    videoPath= os.path.join(videoFolder,'sheepForce={}_wolfForce={}_masterforce={}_masterPullForce={}_masterPullDistanceForSheep={}.avi'.format(sheepForce,wolfForce,masterForce,masterPullForce,masterPullDistanceForSheep))

    # videoPath= os.path.join(mainDemoFolder,'MADDPGMujocoEnvWithRopeAdd2DistractorWithRopePunish_damping={}_frictionloss={}_masterForce={}.avi'.format(damping,frictionloss,masterForce))
    # videoPath= os.path.join(videoFolder,'CrossSheepMADDPGMujocoEnvWithRopeAddDistractor_wolfHideSpeed_damping={}_frictionloss={}_masterForce={}.avi'.format(damping,frictionloss,masterForce))
    # videoPath= os.path.join(videoFolder,'OffsetWolfForwardMADDPGMujocoEnvWithRopeAddDistractor_wolfHideSpeed_damping={}_frictionloss={}_masterForce={}_offsetFrame={}.avi'.format(damping,frictionloss,masterForce,offsetFrame))
    # videoPath = os.path.join(videoFolder,'damping={}_frictionloss={}_masterForce={}_distractorNoise={}.avi'.format(damping,frictionloss,masterForce,distractorNoise))
    # videoPath = os.path.join(videoFolder,'damping={}_frictionloss={}_masterForce={}'.format(damping,frictionloss,masterForce))
    print(videoPath)
    fps = 50
    size=(700,700)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    # fourcc = 0
    videoWriter = cv2.VideoWriter(videoPath,fourcc,fps,size)#最后一个是保存图片的尺寸

    #for(i=1;i<471;++i)

    # pictureFolder = os.path.join(videoFolder,'damping={}_frictionloss={}_masterForce={}'.format(damping,frictionloss,masterForce))
    # pictureFolder = os.path.join(videoFolder,'damping={}_frictionloss={}_masterForce={}_offsetFrame={}'.format(damping,frictionloss,masterForce,offsetFrame))
    # pictureFolder = os.path.join(videoFolder,'damping={}_frictionloss={}_masterForce={}_distractorNoise={}'.format(damping,frictionloss,masterForce,distractorNoise))
    # pictureFolder = os.path.join(videoFolder,'damping={}_frictionloss={}_masterForce={}'.format(damping,frictionloss,masterForce))
    # pictureFolder = os.path.join(videoFolder,'damping={}_frictionloss={}_masterForce={}_offset={}_hideId={}'.format(damping,frictionloss,masterForce,offset,hideId))
    pictureFolder = os.path.join(dataFolder, 'demo', 'MasterForDistractorWolveSpeedUpAndSheepDistancewithFrictionChanged','normal','sheepForce={}_wolfForce={}_masterforce={}_masterPullForce={}_masterPullDistanceForSheep={}'.format(sheepForce,wolfForce,masterForce,masterPullForce,masterPullDistanceForSheep))


    for i in range(0,numTrajToSample*maxRunningStepsToSample):
        imgPath=os.path.join(pictureFolder,'rope'+str(i)+'.png')
        frame = cv2.imread(imgPath)
        img=cv2.resize(frame,size)
        videoWriter.write(img)
    videoWriter.release()
def main():

    manipulatedVariables = OrderedDict()
    # manipulatedVariables['damping'] = [0.5]
    # manipulatedVariables['frictionloss'] = [1.0]
    manipulatedVariables['frictionloss'] = [1.4]
    manipulatedVariables['masterForce'] = [17.0, 19.0]
    manipulatedVariables['killZone'] = [4.0]
    manipulatedVariables['killZoneofDistractor'] = [0.0]
    manipulatedVariables['ropePunishWeight'] = [0.3]
    manipulatedVariables['ropeLength'] = [0.04] #ssr-1,Xp = 0.06; ssr-3 =0.09
    manipulatedVariables['masterMass'] = [1.0] #ssr-1, ssr-3 = 1.0; Xp = 2.0
    manipulatedVariables['masterPunishRange'] = [0.5] #ssr-1, ssr-3 = 1.0; Xp = 2.0
    # manipulatedVariables['wolfMass'] = [3.0/] #ssr-1, ssr-3 = 1.0; Xp = 2.0
    manipulatedVariables['forceAllowedDistance'] = [0.3]
    manipulatedVariables['forceForbiddenPunish'] = [1.0]
    manipulatedVariables['masterPullForce'] = [10.0]
    manipulatedVariables['sheepPunishRange'] = [0.6]
    manipulatedVariables['sheepForce'] = [10.0]
    manipulatedVariables['forceAllowedDistanceForSheep'] = [1.5]
    manipulatedVariables['wolfForce'] = [7.0]
    # manipulatedVariables['offset'] = [0.0]
    # manipulatedVariables['hideId'] = [3]
    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    conditions = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    for condition in conditions:
        print(condition)
        makeVideo(condition)
        # try :
            # makeVideo(condition)
        # except :
            # print('error',condition)
#

if __name__ == '__main__':
    main()
