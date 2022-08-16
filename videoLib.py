'''
/************************************************************************\
$$\       $$$$$$$$\ $$$$$$$\ $$$$$$$$\
$$ |      $$  _____|$$  __$$\\__$$  __|
$$ |      $$ |      $$ |  $$ |  $$ |
$$ |      $$$$$\    $$ |  $$ |  $$ |
$$ |      $$  __|   $$ |  $$ |  $$ |
$$ |      $$ |      $$ |  $$ |  $$ |
$$$$$$$$\ $$ |      $$$$$$$  |  $$ |
\________|\__|      \_______/   \__|

Developed for Laboratory for Fluid Dynamics and Thermodynamics
Bor Zupan
2020 -
\************************************************************************/
#reorganize?
File 'videoLib.py' contains functions for video manipulation, frame extraction and video conversion.
'''
import dataLib as DL

import cv2
import os
import imutils
import shutil
from pathlib import Path
from skimage import data
from skimage.filters import threshold_otsu
from PIL import Image
import numpy as np

def getLength(video_path):
    '''Returns the number of frames in a video'''

    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length


def getFrames(video_path,end_frame=-1,start_frame=0, step=1, BW = False,save_path="output"): #! REPLACES THE OLD getFrames()
    '''
        Extracts .jpg frames from a video

        Input parameters:
            - video_path [string] "path of the video",
            - end_frame [int] "number of frame to stop. Value -1 corresponds to all frames",
            - start_frame [int] "number of frame to start",
            - step [int] "step to extract each n frame",
            - BW [bool] "True/False save in grayscale",
            - save_path [string] "path of the output folder"
    '''

    v_name = DL.extractFileNameFromPath(video_path)
    print(f'Starting frames extraction for {v_name}')

    #create output folder
    Path(save_path).mkdir(parents=True, exist_ok=True)

    _,file_name = os.path.split(video_path) #C:\temp\...\video.mp4 -> C:\temp\...\ , video.mp4
    file_name_without_ext = os.path.splitext(file_name)[0] #video.mp4 -> video

    #check if video is empty
    length = getLength(video_path)
    if length==0:
        print("ERROR : Length of video is 0. Exiting...")
        return 0

    cap = cv2.VideoCapture(video_path)

    #check how many frames we want to extract
    if (end_frame==-1) : count_limit = length - 1
    elif (end_frame != -1 and end_frame < 1) : return print('Error setting end frame. Set to -1 for all or else keep between (1,no_frames)')
    else : count_limit = end_frame

    try:
        count = 0
        skipped = False
        ret = True

        while (ret and count_limit >=count):
            ret, frame = cap.read()
            if ret and count == start_frame: skipped = True
            if skipped:
                if BW: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(os.path.join(
                    save_path,
                    file_name_without_ext+
                    '_{:04d}.jpg'.format(count)
                    #'_no{}.jpg'.format(count)
                ),frame)
                count += 1 * step
                #print(count)
            else:
                count += 1 * step
                #print(count)
    except:
        print("ERROR : Problem with saving test frame cv2 encoding, cannot save file.")
        return 0

    cap.release()

    print("Extracting frames was successful.")


def convertToMp4(files, rotation = 0, BW = False):
    '''
        Convert video from any format to .mp4

        Input parameters:
            - files [array(string)] "array of paths to specified video files",
            - rotation [int] "rotation of the frames in the clockwise direction"
            - BW [bool] "True/False convert video to grayscale"
    '''

    _l = len(files)
    count = 1
    for video in files:
        old_name = video
        print('\n\nConverting video: '+old_name+'  ['+str(count)+'/'+str(_l)+']\n')
        print('Decomposing original video...')
        getFrames(video,BW = BW, save_path='__temp__/')

        images = [f for f in os.listdir('__temp__/') if os.path.isfile(os.path.join('__temp__/', f))]

        print('Reading images...')

        img=[]
        for i in images:
            _ = cv2.imread('__temp__/'+str(i))
            if rotation!= 0: _ = imutils.rotate_bound(_, angle=rotation)
            img.append(_)


        height,width,layers=img[0].shape
        print('Writing .mp4 video...')
        print('Height : Width ',height, width)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        name = video[0:-4]+'.mp4'
        video=cv2.VideoWriter(name,fourcc,10,(width,height))

        for j in img:
            video.write(j)

        cv2.destroyAllWindows()
        video.release()

        print('Deleting temporary files...')
        try:
            shutil.rmtree('__temp__/')
            print('Temporary files deleted.')
        except OSError as e:
            print("Error: %s : %s" % ('__temp__/', e.strerror))

        print('Successfully converted '+old_name+' to '+name+'.')
        count += 1

def importImage(path):
    '''Returns grayscale object PIL.Image from given path'''

    return Image.open(path).convert("L")

def convertToBinary(images):
    '''
        Converts PIL.Image object to binary images

        Input parameters:
            - images [array(PIL.Image)] "list of PIL.Image object images"
        Return parameters:
            - output_bin_arr [array( array(int), array(int), ... )] "list of binary images"
    '''

    output_bin_arr = []
    for img in images:
        opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        opencvImage = cv2.cvtColor(opencvImage, cv2.COLOR_BGR2GRAY)
        b_thresh = threshold_otsu(opencvImage)
        img_bin = (opencvImage >= b_thresh).astype(float).astype(int) * 255
        output_bin_arr.append(img_bin)

    return output_bin_arr


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
#!--------------------------------------------------------------------------------------------OUT-OF-DATE--------------------------------------------------------------------------------------------!#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

def convertFileToBinary(files, background = '', output = 'binary_frames', returnVideo = False):
    '''
        #!FAULTY DOES NOT SAVE FILES!!!
        Extracts frames from video files and binarizes them.

        Input parameters:
            - files [array(string)] "array of paths to specified video files",
            - background [string] "path to the background video file"
            - output [string] "path to the output folder"
            - returnVideo [bool] "True/False composse frames back into .mp4 format"
        Return parameters:
            - fnames [array(string)] "paths to folders of binarized frames"
    '''

    cc = 0
    fnames = []
    if background != '':
        print('Processing background image...')
        getFrames(background,BW = True, save_path='__temp__/')
        b_images = [f for f in os.listdir('__temp__/') if os.path.isfile(os.path.join('__temp__/', f))]
        b_frame = cv2.imread('__temp__/'+str(b_images[0]))
        b_frame = cv2.cvtColor(b_frame, cv2.COLOR_BGR2GRAY)
        b_thresh = threshold_otsu(b_frame)
        BG = (b_frame >= b_thresh).astype(float).astype(int) * 255 #stores background image
        shutil.rmtree('__temp__/')
        print('Background image processed!')
        cc = 1

    if returnVideo: Path('binary_video').mkdir(parents=True, exist_ok=True) #create output folder

    _l = len(files)
    count = 1
    for video in files:
        if video == background: break
        print('\n\nConverting video: '+video+'  ['+str(count)+'/'+str(_l-cc)+']\n')

        getFrames(video,BW = True, save_path='__temp__/')

        print('Reading images...')

        images = [f for f in os.listdir('__temp__/') if os.path.isfile(os.path.join('__temp__/', f))]

        print('Converting images...')

        fname = output + '_' + video[0:-4] + '/'
        fnames.append(fname)
        Path(fname).mkdir(parents=True, exist_ok=True) #create output folder

        count1 = 0
        for i in images:
            frame = cv2.imread('__temp__/'+str(i))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            thresh = threshold_otsu(frame)
            binary = (frame >= thresh).astype(float).astype(int) * 255

            if background != '': binary = 255 - (BG - binary)

            iname = fname + video[0:-4] + '_{:04d}.jpg'.format(count1)
            cv2.imwrite(iname,binary)
            count1 += 1

        if returnVideo:
            print('Building .mp4 binary video')
            images_b = [f for f in os.listdir(fname) if os.path.isfile(os.path.join(fname, f))]
            img=[]
            for i in images_b:
                _ = cv2.imread(fname+str(i))
                img.append(_)


            height,width,layers=img[0].shape
            print('Writing .mp4 video...')
            print('Height : Width ',height, width)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            name = video[0:-4]+'_binary.mp4'
            str_video = video
            video=cv2.VideoWriter(name,fourcc,10,(width,height))

            for j in img:
                video.write(j)

            cv2.destroyAllWindows()
            video.release()

            shutil.move(name, 'binary_video/'+name)

            print('Successfully converted '+str_video+' to binary.')

        print('Deleting temporary files...')
        shutil.rmtree('__temp__/')
        print('Temporary files deleted.')

        count += 1
    return fnames

def getFrames_OLD(video_path,no_frames=-1,skip_frames=1,BW = False,save_path="output"): #! THIS FUNCTION IS OUT-OF-DATE. IT HAS BEEN REPLACED BY getFrames()
    '''
        Extracts .jpg frames from a video

        Input parameters:
            - video_path [string] "path of the video",
            - no_frames [int] "number of frames to extract. Value -1 corresponds to all frames",
            - skip_frames [int] "number of frames to skip",
            - BW [bool] "True/False save in grayscale",
            - save_path [string] "path of the output folder"
    '''

    #create output folder
    Path(save_path).mkdir(parents=True, exist_ok=True)

    _,file_name = os.path.split(video_path) #C:\temp\...\video.mp4 -> C:\temp\...\ , video.mp4
    file_name_without_ext = os.path.splitext(file_name)[0] #video.mp4 -> video

    #check if video is empty
    length = getLength(video_path)
    if length==0:
        print("ERROR : Length of video is 0. Exiting...")
        return 0

    cap = cv2.VideoCapture(video_path)
    count = 0 #kep count the number of frames

    #check how many frames we want to extract
    if (no_frames==-1) : count_limit = length
    elif (no_frames != -1 and no_frames < 1) : count_limit = 1
    else : count_limit = no_frames

    #test first frame
    ret, frame = cap.read()
    test_file_path = os.path.join(
        save_path,
        file_name_without_ext+ \
        '_{:04d}.jpg'.format(count)
        #'_no{}.jpg'.format(count)
    )

    if BW: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(test_file_path,frame)
    if os.path.isfile(test_file_path):

        print("Saving test frame was successful. Proceeding...")

        count = 1

        while (ret and count_limit >=count):
            ret, frame = cap.read()
            if ret and count % skip_frames == 0:
                if BW: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(os.path.join(
                    save_path,
                    file_name_without_ext+
                    '_{:04d}.jpg'.format(count)
                    #'_no{}.jpg'.format(count)
                ),frame)
                count += 1
                #print(count)
            else:
                count += 1
                #print(count)
    else:
        print("ERROR : Problem with saving test frame cv2 encoding, cannot save file.")
        return 0

    cap.release()

    print("Extracting frames was successful.")