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
Krištof Kovačič, Bor Zupan
Updated for Laboratory for Fluid Dynamics and Thermodynamics
Krištof Kovačič
2021 -
\************************************************************************/

This file contains functions for measuring jet parameters in the nozzle experiment at LFDT.
'''

import videoLib as VL
import plotLib as PL
import dataLib as DL

import sys
import os
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from skimage import measure
import cv2
import PIL.ImageChops as IC
from statistics import multimode
from itertools import groupby



"---------------------------------------NUMERICAL LIBRARY FOR VIDEO POST-PROCESSING-----------------------------------"

def calcRefNum(reference, scale):
    '''
    @author: kristof
    
        Calculates the mm per pixel ratio for CFD Post procesing.

        Input parameters:
            - reference [Image path] "reference image",
            - scale [float] "size of the reference object in mm"
            - showImage [bool] "True/False display the measured region"
        Return parameters:
            - ratio [float] "mm/px ratio. Multiply this ratio to any given px value to calculate the true value in mm"
            - px_diam [float] diameter of the reference object in px"
    '''
    ref = VL.importImage(reference)

    ref_bin = VL.convertToBinary([ref])

    img_ref = np.asarray(ref_bin)


    im = Image.open(reference)
    width, height = im.size
        
    "measuring width [pixels] of the nozzle outlet diameter, which is a reference object, in the first line of the binarized reference frame [PIL.Image]"
    start = []
    end = []
    for i in range(0, width-2):
        if (img_ref[0][0][i] != img_ref[0][0][i+1]) and (img_ref[0][0][i+1] == img_ref[0][0][i+2]) and img_ref[0][0][i+1]<255:
            start.append(i+1)
        if (img_ref[0][0][i] == img_ref[0][0][i+1]) and (img_ref[0][0][i+1] != img_ref[0][0][i+2]) and img_ref[0][0][i+1]<255:
            end.append(i+1)
    
    
    if len(start)>1:
        start = [start[int(len(start)/2)]]
    else:
        start = start[0]
    if len(end)>1:
        end = [end[int(len(end)/2)]]
    else:
        end = end[0]
        
    px_diam = end-start + 1
    ratio = scale / px_diam
    print(ratio, px_diam)
    

    return (ratio, px_diam)


def defineStartJetNum(reference, first_n_lines=100, logotype_frac=0.9 ):
    '''
    @author: kristof
    
        Defines a start of the jet, which is the outlet of the nozzle's orifice for CFD Post procesing.

        Input parameters:
            - reference [Image path] "reference image, which should be colored, not grayscale!",
            - first_n_lines [int] "nr. of lines to skip from first line"
            - logotype_frac [float] "fraction of whole picture's width, to crop the logotype"
        Return parameters:
            - define start [float] "number of line where jet start, outside the nozzle"
            
    '''     
       
    ref = VL.importImage(reference)
     
    ref_bin =VL.convertToBinary([ref])
     
    #cut the Ansys logotype
    pic = Image.open(reference)
    width, height = pic.size
       
    im_a = ref_bin[0]
    im_b = im_a[:, 0:int(logotype_frac * width)]
    
    
    "measuring width [pixels] of the nozzle geometry, in the first 30 lines of the binarized reference frame [PIL.Image]"
    
    geom_width = []
    
    for j in range(0, first_n_lines, 1):
        start = []
        end = []
        for i in range(0, int(logotype_frac * width)-2):
            if (im_b[j][i] != im_b[j][i+1]) and (im_b[j][i+1] == im_b[j][i+2]) and im_b[j][i+1]<255:
                start.append(i+1)
            if (im_b[j][i] == im_b[j][i+1]) and (im_b[j][i+1] != im_b[j][i+2]) and im_b[j][i+1]<255:
                end.append(i+1)
            
        if len(start)>1:
            start = [start[int(len(start)/2)]]
        else:
            start = start[0]
        if len(end)>1:
            end = [end[int(len(end)/2)]]
        else:
            end = end[0]
                 
        _gw = end-start
        geom_width.append(_gw)
    
    define_start = []
    for i in range(0, len(geom_width)-1, 1):
        if (geom_width[i+1] / geom_width[i] < 0.5 or geom_width[i+1] / geom_width[i] > 1.5): 
            #1 +- 0.5 should be smaller than orifice diameter to external chamber diameter ratio
            define_start.append(i+1) #i+1 because is the first line below nozzle, where jet starts
    
    print("Line of the jet start in px: ", define_start[0])
    
    return(define_start[0])

def measureJetFrameNum(path, frame, outlet_start, locations_px, left_cut = 400, right_cut = 1300, saveImage = False, outputf = 'measured_lines/'):
    '''
    @author: kristof
    
        This function extracts jet parameters from a .jpg frame of a jet.

        Input parameters:
            - path [string] "path of the frame",
            - frame [PIL.Image] "Image object of the frame, binary PIL.Image",
            - outlet_start [line number] "line where jet starts, use defineStartJetNum function ", 
            - locations_px [list(float)] "list of locations (in px) at which the jet diameters wanted to be checked"
            - left_cut [column nr.] "column number of the frame to cut the image, e.g. logotypes, scales, axis etc.",
            - right_cut [column nr.] "column number of the frame to cut the image, e.g. logotypes, scales, axis etc.",
            - saveImage [bool] "True/False save image with calculated parameters",
            - outputf [string] "path to output folder"
        Return parameters:
            - d0 [float] "average diameter of the jet with correction for skewness",
            - l0 [float] "average length of the jet with correction for skewness",
            - np.degrees(phi) [float] "skewness angle in degrees",
            - threshold_d [float] "threshold diameter calculated from lines 10:20 of the jet",
            - jet_line_start [int] "frame line (from [0,height]) where jet starts",
            - jet_line_end [int] "frame line (from [0,height]) where jet ends",
            - jet_line_end_stable [int] "frame line (from [0,height]) where stable jet ends",
            - jet_arr [array((array(int),int))] "array of image line pixels (values from (0,255)) and frame line height ",
            - jet_d_arr [array((int,int,int))] "array of tuples where the [0] stores the number of pixels across the jet, [1] frame line height and [2] number of pixels to the beginning of the jet from the left",
            - jet_d_arr_stable [array((int,int,int))] "stores the same values as jet_d_arr but only to the end of the stable jet defined by breakRatio",
            - stable_d0 [float] "diameter of the stable jet with correction for skewness"
    '''
    
    #this block defines start of the jet, which is nozzle's outlet line in the image      
    #outlet_start = defineStartJetNum(reference)
    
    
    #cuting image, e.g. logotypes, scales, axis, etc.  
    img = np.asarray(frame[:, left_cut:right_cut])
    
    width = len(img[0])
    jet_line_start = 0
    jet_line_end = 0
    
    #this block gets an array of line pixel values over the whole jet without droplets
    jet_arr = [] #include frame lines of jet
    count = -1
    check_jet_start = False #True when jet startes
    for line in img: #loop through the frame pixel lines and extract lines that include the jet
        count += 1
        if np.average(line)  != 255: check_jet_start = True
        if np.average(line) == 255 and check_jet_start: break
        #if np.average(line) > 255*0.99: continue
        jet_arr.append((line,count))
    
    #this block takes into account the start of the jet at the nozzle's outlet
    jet_arr = jet_arr[outlet_start:]
    
        
    #this block calculates the average diameter of the first n lines of the jet
    threshold_d_arr = [] #list of first n line jet diameters starting from mth line (to remove first unstable lines)
    for line in jet_arr[10:20]: #10:20
        line = line[0]
        _jet = False #True when jet starts
        count_px = 0 #counts the number of pixel across the jet diameter
        for i in range(0,width):
            if np.all(line[i:i+5] < 10): _jet = True
            if _jet and line[i] > 10: break
            if _jet: count_px += 1
        if count_px < 50: threshold_d_arr.append(count_px) #if the nozzle in the background was to high and there is a thick line at the top
    threshold_d = np.average(threshold_d_arr) #average diameter of first 10 lines    
    
    #this block calculates the diameter of the jet.
    jet_d_arr = [] #array of jet diameters
    for line in jet_arr:
        line_num = line[1]
        line = line[0]
        _jet = False #True when jet starts
        count_px = 0 #counts the number of pixel across the jet diameter
        count_px_left = 0 #counts the number of pixel before jet starts
        for i in range(0,width):
            if np.all(line[i:i+5] < 10): _jet = True
            if _jet and line[i] > 10: break
            if _jet: count_px += 1
            else: count_px_left += 1
        jet_d_arr.append((count_px, line_num, count_px_left))
    
    #This block calculates the diameter of the jet at desired locations
    jet_d_loc = []
    for i in range(0, len(locations_px), 1):
        for j in range(0, len(jet_d_arr), 1):
            if locations_px[i] == jet_d_arr[j][1]:
                jet_d_loc.append(jet_d_arr[j][0:-1])
    
        if locations_px[i] > jet_d_arr[-1][1]:
            jet_d_loc.append((0, locations_px[i]))
    
    #remove first unstable values unless the frame has no nozzle trace. check for trace with standard deviation of the first 15 jet diameters (if there is none it should be under 1)
    first_array = [i[0] for i in jet_d_arr[0:15]]
    first_array_std = np.std(first_array)
    
    if first_array_std > 3:
        _pos = 0
        _koef = 5 #koeficient, 5
        for i in range(0,20):
            if np.abs(jet_d_arr[20-i][0] - threshold_d) > _koef: _pos = i; break
        jet_d_arr = jet_d_arr[21-_pos:]
        jet_line_start += jet_d_arr[0][1]
    
    
    #set where jet structure ends
    jet_line_end = jet_d_arr[-1][1]
    
    
    #check end of stable jet by variance of the diameter
    jet_d_arr_stable = jet_d_arr[0:10]
    
    jet_d_arr_stable = jet_d_arr[0:10]
    _len = len(jet_d_arr)
    for i in range(10,_len):
        var_arr = jet_d_arr[i-4:i+1] #i-5
        var_arr = [i[0] for i in var_arr]
        var = np.var(var_arr)
        #print(i,var)
        if var>1 : break
        jet_d_arr_stable.append(jet_d_arr[i])
    
    jet_line_end_stable = jet_d_arr_stable[-1][1]   
    
    #calculate average diameter of the stable jet
    stable_d_arr = []
    for values in jet_d_arr_stable:
        stable_d_arr.append(values[0])
    avg_stable_d_px = np.average(stable_d_arr)
    
    
    #calculate average jet diameter and jet length in case jet is straight
    jet_d = [i[0] for i in jet_d_arr]
    jet_avg_d_px = np.average(jet_d)
    jet_l_px = jet_d_arr[-1][1] - jet_d_arr[0][1]
    
    
    #calculate jet skewness
    L1 = (jet_d_arr[0][2], jet_d_arr[0][1])
    R1 = (jet_d_arr[0][2] + jet_d_arr[0][0], jet_d_arr[0][1])
    L2 = (jet_d_arr_stable[-1][2], jet_d_arr_stable[-1][1])
    R2 = (jet_d_arr_stable[-1][2] + jet_d_arr_stable[-1][0], jet_d_arr_stable[-1][1])
    M1 = (int(np.ceil(0.5*(L1[0] + R1[0]))), jet_d_arr_stable[0][1])
    M2 = (int(np.ceil(0.5*(L2[0] + R2[0]))), jet_d_arr_stable[-1][1])
    L3 = (jet_d_arr[-1][2]-10, jet_d_arr[-1][1])
    R3 = (jet_d_arr[-1][2] + jet_d_arr[-1][0]+10, jet_d_arr[-1][1])
    STABLE1 = (M2[0]-10,M2[1])
    STABLE2 = (M2[0]+10,M2[1])
    
    
    #calculate true average diameter and length
    sk = M2[0] - M1[0]
    l = jet_l_px
    d = jet_avg_d_px
    phi = np.arctan(sk/l)
    l0 = l/np.cos(phi)
    d0 = d*np.cos(phi)
    stable_d0 = avg_stable_d_px*np.cos(phi) 
    
    #make image
    if saveImage:
        _,file_name = os.path.split(path)
        Path(outputf).mkdir(parents=True, exist_ok=True)
        PL.drawLines(path,output = outputf+file_name,points = [(L1[0], L1[1], R1[0], R1[1]), (L3[0], L3[1], R3[0], R3[1]), (M1[0], M1[1], M2[0], M2[1]), (STABLE1[0], STABLE1[1], STABLE2[0], STABLE2[1])],color = (255,0,0),thickness = 1) #outputs .jpg with marked start od end of the stable jet
    
    #this block takes into account the start of jet at the nozzle's outlet line
    jet_line_start = outlet_start + jet_line_start    

    #return arguments
    return (d0, l0, np.degrees(phi), threshold_d, jet_line_start, jet_line_end, jet_line_end_stable, jet_arr, jet_d_arr, jet_d_arr_stable,stable_d0, jet_d_loc)


def measureJetVideoNum(video_path, bcg_path, ref_path, scale, locations, start_frame = 0, end_frame = -1, left_cut = 0, right_cut = 1776, note=''):
    '''
    @author: kristof
    
        Measures jet parameters over a whole video.

        Input parameters:
            - video_path [string] "path to the video",
            - ref_path [string] "path to the reference image/frame, !not video! for defining nozzle's outlet, !should be colored, not grayscale!"
            - scale "reference object in mm (located at the first line of the image, normally nozzle's outlet orifice diameter)"
            - locations [list(float)] "list of locations (in mm or other working unit) at which the jet diameters wanted to be checked"
            - start_frame [int] "select starting frame",
            - end_frame [int] "select ending frame",
            - left_cut [column nr.] "column number of the frame to cut the image, e.g. logotypes, scales, axis etc.",
            - right_cut [column nr.] "column number of the frame to cut the image, e.g. logotypes, scales, axis etc.",
            - note [string] "add note to file name for AVG and ALL data"
        Return parameters:
            - G [int] "gas volume flow in l/h",
            - L [int] "liquid volume flow in ul/min",
            - avg_jet_d [float] "average jet diameter in mm across the frames",
            - avg_jet_l [float] "average jet length in mm across the frames",
            - avg_jet_phi [float] "average jet skewness in degrees across the frames",
            - ref_ratio [float] "mm/px ratio"
            - d_min [float] "minimal jet diameter in mm",
            - d_max [float] "maximal jet diameter in mm",
            - l_min [float] "minimal jet length in mm",
            - l_max [float] "maximal jet length in mm",
            - d_std [float] "diameter standard deviation in mm",
            - l_std [float] "length standard deviation in mm",
            - avg_jet_stable_d [float] "average stable jet diameter in mm across the frames in mm",
            - stable_d_min [float] "minimal stable jet diameter in mm",
            - stable_d_max [float] "maximal stable jet diameter in mm"
    '''

    #extract frames in temp folders, import them as PIL.Image objects and delete temp files
    print('Starting calculation for '+video_path)
    print('Preparing frames...')
    VL.getFrames(video_path, end_frame=end_frame, start_frame=start_frame, BW = True, save_path='_temp_video_frames')
    VL.getFrames(bcg_path, BW = True, save_path='_temp_bcg_frames')
    VL.getFrames(ref_path, BW = True, save_path='_temp_ref_frames')

    video_frames_path = DL.getFiles(path = '_temp_video_frames/', extension = '.jpg')
    bcg_frame_path = DL.getFiles(path = '_temp_bcg_frames/', extension = '.jpg')
    ref_frame_path = ref_path

    video_frames_arr = []
    bcg_frame = VL.importImage(bcg_frame_path[0])
    #ref_frame = VL.importImage(ref_frame_path[0])

    for path in video_frames_path:
        video_frames_arr.append(VL.importImage(path))

    print('Deleting temporary files...')
    shutil.rmtree('_temp_video_frames/')
    shutil.rmtree('_temp_bcg_frames/')
    #shutil.rmtree('_temp_ref_frames/')
    print('Temporary files deleted.')


    #substract background from jet and binarize frames
    print('Substracting background and binarizing frames...')
    sub_jet_frames_arr = []
    for i in video_frames_arr:
        _ = IC.subtract(i, bcg_frame, offset=255)
        _bin = VL.convertToBinary([_])
        sub_jet_frames_arr.append(_bin[0])


    #calculate mm/px ratio
    print('Calculating mm/px ratio...')
    #_ref_sub = IC.subtract(ref_frame, bcg_frame, offset=255)
    #ref_bin_frame = VL.convertToBinary([_ref_sub])
    ref_ratio = calcRefNum(ref_path, scale)[0]
    
    
    #defines nozzle's outlet line on image
    print("Calculating nozzle's outlet line...")
    outlet_start = defineStartJetNum(ref_path, first_n_lines=50, logotype_frac=0.9)
    
    #calculate locations from mm to px
    #locations = [0, 0.5, 1.0]
    locations_px = [round(x / ref_ratio + outlet_start) for x in locations] 
    
    #define output arrays
    avg_d_arr = []
    avg_l_arr = []
    phi_arr = []
    stable_d_arr = []
    loc_d = []
    
    #measure jet for each frame
    print('Calculating jet parameters for each frame... (this will take a while)')
    count_frames = np.arange(0,len(sub_jet_frames_arr),1)
    for frame,i in zip(sub_jet_frames_arr,count_frames):
        try:
            if i%50 == 0: 
                sys.stdout.write(f'Progress {np.ceil((i/len(sub_jet_frames_arr))*100)}%         \r')
                sys.stdout.flush()
            data = measureJetFrameNum('', frame, outlet_start, locations_px, left_cut, right_cut)
            if data[0] > 0: #in case if some frame is corrupted, e.g. just white, without a jet
                avg_d_arr.append(data[0])
                avg_l_arr.append(data[1])
                phi_arr.append(data[2])
                stable_d_arr.append(data[10])
                loc_d.append(data[11])
        except:
            print('Error. Frame {}/{} corrupted'.format(i,len(sub_jet_frames_arr)))
            sys.stdout.write(f'Progress {np.ceil((i/len(sub_jet_frames_arr))*100)}%         \r')
            sys.stdout.flush()
    print('Calculating jet parameters for each frame COMPLETED.')
    
    
    #calculate jet parameters in mm and degrees
    avg_d_arr_mm = [i* ref_ratio for i in avg_d_arr]
    avg_l_arr_mm = [i* ref_ratio for i in avg_l_arr]
    stable_d_arr_mm = [i* ref_ratio for i in stable_d_arr]
    avg_jet_d = np.average(avg_d_arr_mm)
    avg_jet_l = np.average(avg_l_arr_mm)
    avg_jet_phi = np.average(phi_arr)
    avg_jet_stable_d = np.average(stable_d_arr_mm)
    d_min = np.min(avg_d_arr_mm)
    d_max = np.max(avg_d_arr_mm)
    l_min = np.min(avg_l_arr_mm)
    l_max = np.max(avg_l_arr_mm)
    d_std = np.std(avg_d_arr_mm)
    l_std = np.std(avg_l_arr_mm)
    stable_d_min = np.min(stable_d_arr_mm)
    stable_d_max = np.max(stable_d_arr_mm)
    
    d_location = []
    loc_name = []
    for i in range(0, len(loc_d), 1):
        dl = []
        for j in range(0, len(loc_d[i]), 1):
            dl.append(ref_ratio * loc_d[i][j][0])
            
            if i == 0:
                loc_name.append(round(ref_ratio * (loc_d[i][j][1] - outlet_start), 4) )
        d_location.append(dl)
        
    #write data
    print('Writing data to .txt file...')
    out_d = ['average jet diameter [mm]',avg_jet_d]
    out_l = ['average jet length [mm]',avg_jet_l]
    out_phi = ['average jet skewness [deg]',avg_jet_phi]
    out_ratio = ['mm to px ratio [mm/px]',ref_ratio]
    out_d_min = ['Minimal d [mm]', d_min]
    out_d_max = ['Maximal d [mm]', d_max]
    out_l_min = ['Minimal l [mm]', l_min]
    out_l_max = ['Maximal l [mm]', l_max]
    out_d_std = ['Standard deviation of d [mm]', d_std]
    out_l_std = ['Standard deviation of l [mm]', l_std]
    out_stable_d = ['average stable jet diameter [mm]',avg_jet_stable_d]
    out_stable_d_min = ['Minimal stable d [mm]', stable_d_min]
    out_stable_d_max = ['Maximal stable d [mm]', stable_d_max]
    fname = video_path[0:-4] + '_AVG_data'+note+'.txt'
    DL.save2DDataToTxt([out_d,out_l,out_phi,out_ratio,out_d_min,out_d_max,out_l_min,out_l_max,out_d_std,out_l_std,out_stable_d,out_stable_d_min,out_stable_d_max], filename=fname, transpose = True)
    
    out_d_frame = ['average jet diameter by frame [mm]'] + avg_d_arr_mm
    out_l_frame = ['average jet length by frame [mm]'] + avg_l_arr_mm
    out_phi_frame = ['average jet skewness by frame [deg]'] + phi_arr
    out_stable_d_frame = ['average stable jet diameter by frame [mm]'] + stable_d_arr_mm
    fname_frame = video_path[0:-4] + '_ALL_data'+note+'.txt'
    DL.save2DDataToTxt([out_d_frame,out_l_frame,out_phi_frame,out_stable_d_frame], filename=fname_frame, transpose = True)
    
    
    out_loc_d_frame = [['jet diameter at location ' + str(x) for x in loc_name]] + d_location
    fname_loc = video_path[0:-4] + '_location_data'+note+'.txt'
    DL.save2DDataToTxt(out_loc_d_frame, filename=fname_loc, transpose = False)
    
    
    
    #extract flow parameters
    head, tail = os.path.split(video_path)
    video_name = tail[:-4]
    G = video_name[1:3]
    L = video_name[4:]
    
    
    #return data
    print(f'Calculation for video {video_path} COMPLETE!')
    print('\n')
    out = (G,L,avg_jet_d, avg_jet_l, avg_jet_phi, ref_ratio,d_min,d_max,l_min,l_max,d_std,l_std,avg_jet_stable_d,stable_d_min,stable_d_max,d_location)
    return out


def measureJetVideoNum_brez_ozadja(video_path, ref_path, scale, locations, start_frame = 0, end_frame = -1, left_cut = 0, right_cut = 1776, note=''):
    '''
    @author: kristof
    
        Measures jet parameters over a whole video.

        Input parameters:
            - video_path [string] "path to the video",
            - ref_path [string] "path to the reference image/frame, !not video! for defining nozzle's outlet, !should be colored, not grayscale!"
            - scale "reference object in mm (located at the first line of the image, normally nozzle's outlet orifice diameter)"
            - start_frame [int] "select starting frame",
            - locations [list(float)] "list of locations (in mm or other working unit) at which the jet diameters wanted to be checked"
            - end_frame [int] "select ending frame",
            - left_cut [column nr.] "column number of the frame to cut the image, e.g. logotypes, scales, axis etc.",
            - right_cut [column nr.] "column number of the frame to cut the image, e.g. logotypes, scales, axis etc.",
            - note [string] "add note to file name for AVG and ALL data"
        Return parameters:
            - G [int] "gas volume flow in l/h",
            - L [int] "liquid volume flow in ul/min",
            - avg_jet_d [float] "average jet diameter in mm across the frames",
            - avg_jet_l [float] "average jet length in mm across the frames",
            - avg_jet_phi [float] "average jet skewness in degrees across the frames",
            - ref_ratio [float] "mm/px ratio"
            - d_min [float] "minimal jet diameter in mm",
            - d_max [float] "maximal jet diameter in mm",
            - l_min [float] "minimal jet length in mm",
            - l_max [float] "maximal jet length in mm",
            - d_std [float] "diameter standard deviation in mm",
            - l_std [float] "length standard deviation in mm",
            - avg_jet_stable_d [float] "average stable jet diameter in mm across the frames in mm",
            - stable_d_min [float] "minimal stable jet diameter in mm",
            - stable_d_max [float] "maximal stable jet diameter in mm"
    '''

    #extract frames in temp folders, import them as PIL.Image objects and delete temp files
    print('Starting calculation for '+video_path)
    print('Preparing frames...')
    VL.getFrames(video_path, end_frame=end_frame, start_frame=start_frame, BW = True, save_path='_temp_video_frames')
    #VL.getFrames(bcg_path, BW = True, save_path='_temp_bcg_frames')
    VL.getFrames(ref_path, BW = True, save_path='_temp_ref_frames')
    
    video_frames_path = DL.getFiles(path = '_temp_video_frames/', extension = '.jpg')
    #bcg_frame_path = DL.getFiles(path = '_temp_bcg_frames/', extension = '.jpg')
    ref_frame_path = ref_path
    
    video_frames_arr = []
    #bcg_frame = VL.importImage(bcg_frame_path[0])
    #ref_frame = VL.importImage(ref_frame_path[0])
    
    for path in video_frames_path:
        video_frames_arr.append(VL.importImage(path))
    
    print('Deleting temporary files...')
    shutil.rmtree('_temp_video_frames/')
    #shutil.rmtree('_temp_bcg_frames/')
    #shutil.rmtree('_temp_ref_frames/')
    print('Temporary files deleted.')
    
    
    #substract background from jet and binarize frames
    #print('Substracting background and binarizing frames...')
    sub_jet_frames_arr = []
    for i in video_frames_arr:
        _ = i
        #_ = IC.subtract(i, bcg_frame, offset=255)
        _bin = VL.convertToBinary([_])
        sub_jet_frames_arr.append(_bin[0])
    
    
    #calculate mm/px ratio
    print('Calculating mm/px ratio...')
    #_ref_sub = IC.subtract(ref_frame, bcg_frame, offset=255)
    #ref_bin_frame = VL.convertToBinary([_ref_sub])
    ref_ratio = calcRefNum(ref_path, scale)[0]
    
    
    #defines nozzle's outlet line on image
    print("Calculating nozzle's outlet line...")
    outlet_start = defineStartJetNum(ref_path, first_n_lines=50, logotype_frac=0.9)
    
    #calculate locations from mm to px
    #locations = [0, 0.5, 1.0]
    locations_px = [round(x / ref_ratio + outlet_start) for x in locations] 
    
    #define output arrays
    avg_d_arr = []
    avg_l_arr = []
    phi_arr = []
    stable_d_arr = []
    loc_d = []
    
    #measure jet for each frame
    print('Calculating jet parameters for each frame... (this will take a while)')
    count_frames = np.arange(0,len(sub_jet_frames_arr),1)
    for frame,i in zip(sub_jet_frames_arr,count_frames):
        try:
            if i%50 == 0: 
                sys.stdout.write(f'Progress {np.ceil((i/len(sub_jet_frames_arr))*100)}%         \r')
                sys.stdout.flush()
            data = measureJetFrameNum('', frame, outlet_start, locations_px, left_cut, right_cut)
            if data[0] > 0: #in case if some frame is corrupted, e.g. just white, without a jet
                avg_d_arr.append(data[0])
                avg_l_arr.append(data[1])
                phi_arr.append(data[2])
                stable_d_arr.append(data[10])
                loc_d.append(data[11])
        except:
            print('Error. Frame {}/{} corrupted'.format(i,len(sub_jet_frames_arr)))
            sys.stdout.write(f'Progress {np.ceil((i/len(sub_jet_frames_arr))*100)}%         \r')
            sys.stdout.flush()
    print('Calculating jet parameters for each frame COMPLETED.')
    
    
    #calculate jet parameters in mm and degrees
    avg_d_arr_mm = [i* ref_ratio for i in avg_d_arr]
    avg_l_arr_mm = [i* ref_ratio for i in avg_l_arr]
    stable_d_arr_mm = [i* ref_ratio for i in stable_d_arr]
    avg_jet_d = np.average(avg_d_arr_mm)
    avg_jet_l = np.average(avg_l_arr_mm)
    avg_jet_phi = np.average(phi_arr)
    avg_jet_stable_d = np.average(stable_d_arr_mm)
    d_min = np.min(avg_d_arr_mm)
    d_max = np.max(avg_d_arr_mm)
    l_min = np.min(avg_l_arr_mm)
    l_max = np.max(avg_l_arr_mm)
    d_std = np.std(avg_d_arr_mm)
    l_std = np.std(avg_l_arr_mm)
    stable_d_min = np.min(stable_d_arr_mm)
    stable_d_max = np.max(stable_d_arr_mm)
    
    d_location = []
    loc_name = []
    for i in range(0, len(loc_d), 1):
        dl = []
        for j in range(0, len(loc_d[i]), 1):
            dl.append(ref_ratio * loc_d[i][j][0])
            
            if i == 0:
                loc_name.append(round(ref_ratio * (loc_d[i][j][1] - outlet_start), 4) )
        d_location.append(dl)
        
    #write data
    print('Writing data to .txt file...')
    out_d = ['average jet diameter [mm]',avg_jet_d]
    out_l = ['average jet length [mm]',avg_jet_l]
    out_phi = ['average jet skewness [deg]',avg_jet_phi]
    out_ratio = ['mm to px ratio [mm/px]',ref_ratio]
    out_d_min = ['Minimal d [mm]', d_min]
    out_d_max = ['Maximal d [mm]', d_max]
    out_l_min = ['Minimal l [mm]', l_min]
    out_l_max = ['Maximal l [mm]', l_max]
    out_d_std = ['Standard deviation of d [mm]', d_std]
    out_l_std = ['Standard deviation of l [mm]', l_std]
    out_stable_d = ['average stable jet diameter [mm]',avg_jet_stable_d]
    out_stable_d_min = ['Minimal stable d [mm]', stable_d_min]
    out_stable_d_max = ['Maximal stable d [mm]', stable_d_max]
    fname = video_path[0:-4] + '_AVG_data'+note+'.txt'
    DL.save2DDataToTxt([out_d,out_l,out_phi,out_ratio,out_d_min,out_d_max,out_l_min,out_l_max,out_d_std,out_l_std,out_stable_d,out_stable_d_min,out_stable_d_max], filename=fname, transpose = True)
    
    out_d_frame = ['average jet diameter by frame [mm]'] + avg_d_arr_mm
    out_l_frame = ['average jet length by frame [mm]'] + avg_l_arr_mm
    out_phi_frame = ['average jet skewness by frame [deg]'] + phi_arr
    out_stable_d_frame = ['average stable jet diameter by frame [mm]'] + stable_d_arr_mm
    fname_frame = video_path[0:-4] + '_ALL_data'+note+'.txt'
    DL.save2DDataToTxt([out_d_frame,out_l_frame,out_phi_frame,out_stable_d_frame], filename=fname_frame, transpose = True)
    
    
    out_loc_d_frame = [['jet diameter at location ' + str(x) for x in loc_name]] + d_location
    fname_loc = video_path[0:-4] + '_location_data'+note+'.txt'
    DL.save2DDataToTxt(out_loc_d_frame, filename=fname_loc, transpose = False)
    
    
    
    #extract flow parameters
    head, tail = os.path.split(video_path)
    video_name = tail[:-4]
    G = video_name[1:3]
    L = video_name[4:]
    
    
    #return data
    print(f'Calculation for video {video_path} COMPLETE!')
    print('\n')
    out = (G,L,avg_jet_d, avg_jet_l, avg_jet_phi, ref_ratio,d_min,d_max,l_min,l_max,d_std,l_std,avg_jet_stable_d,stable_d_min,stable_d_max,d_location)
    return out


def measureJetVideoNum_brez_ozadja_brez_videa(video_path, ref_path, scale, locations, start_frame = 0, end_frame = -1, left_cut = 0, right_cut = 1776, note=''):
    '''
    @author: kristof
    
        Measures jet parameters over a whole video.

        Input parameters:
            - video_path [string] "path to the folder of the frames",
            - ref_path [string] "path to the reference image/frame, !not video! for defining nozzle's outlet, !should be colored, not grayscale!"
            - scale "reference object in mm (located at the first line of the image, normally nozzle's outlet orifice diameter)"
            - locations [list(float)] "list of locations (in mm or other working unit) at which the jet diameters wanted to be checked"
            - start_frame [int] "select starting frame",
            - end_frame [int] "select ending frame",
            - left_cut [column nr.] "column number of the frame to cut the image, e.g. logotypes, scales, axis etc.",
            - right_cut [column nr.] "column number of the frame to cut the image, e.g. logotypes, scales, axis etc.",
            - note [string] "add note to file name for AVG and ALL data"
        Return parameters:
            - G [int] "gas volume flow in l/h",
            - L [int] "liquid volume flow in ul/min",
            - avg_jet_d [float] "average jet diameter in mm across the frames",
            - avg_jet_l [float] "average jet length in mm across the frames",
            - avg_jet_phi [float] "average jet skewness in degrees across the frames",
            - ref_ratio [float] "mm/px ratio"
            - d_min [float] "minimal jet diameter in mm",
            - d_max [float] "maximal jet diameter in mm",
            - l_min [float] "minimal jet length in mm",
            - l_max [float] "maximal jet length in mm",
            - d_std [float] "diameter standard deviation in mm",
            - l_std [float] "length standard deviation in mm",
            - avg_jet_stable_d [float] "average stable jet diameter in mm across the frames in mm",
            - stable_d_min [float] "minimal stable jet diameter in mm",
            - stable_d_max [float] "maximal stable jet diameter in mm"
    '''

    #extract frames in temp folders, import them as PIL.Image objects and delete temp files
    #print('Starting calculation for '+video_path)
    #print('Preparing frames...')
    #VL.getFrames(video_path, end_frame=end_frame, start_frame=start_frame, BW = True, save_path='_temp_video_frames')
    #VL.getFrames(bcg_path, BW = True, save_path='_temp_bcg_frames')
    #VL.getFrames(ref_path, BW = True, save_path='_temp_ref_frames')

    video_frames_path = DL.getFiles(path = video_path, extension = '.jpg')
    video_frames_path = video_frames_path[start_frame:end_frame]
    #bcg_frame_path = DL.getFiles(path = '_temp_bcg_frames/', extension = '.jpg')
    ref_frame_path = ref_path

    video_frames_arr = []
    #bcg_frame = VL.importImage(bcg_frame_path[0])
    #ref_frame = VL.importImage(ref_frame_path[0])

    for path in video_frames_path:
        video_frames_arr.append(VL.importImage(path))

    print('Deleting temporary files...')
    #shutil.rmtree('_temp_video_frames/')
    #shutil.rmtree('_temp_bcg_frames/')
    #shutil.rmtree('_temp_ref_frames/')
    print('Temporary files deleted.')


    #substract background from jet and binarize frames
    #print('Substracting background and binarizing frames...')
    sub_jet_frames_arr = []
    for i in video_frames_arr:
        _ = i
        #_ = IC.subtract(i, bcg_frame, offset=255)
        _bin = VL.convertToBinary([_])
        sub_jet_frames_arr.append(_bin[0])


    #calculate mm/px ratio
    print('Calculating mm/px ratio...')
    #_ref_sub = IC.subtract(ref_frame, bcg_frame, offset=255)
    #ref_bin_frame = VL.convertToBinary([_ref_sub])
    ref_ratio = calcRefNum(ref_path, scale)[0]
    
    
    #defines nozzle's outlet line on image
    print("Calculating nozzle's outlet line...")
    outlet_start = defineStartJetNum(ref_path, first_n_lines=50, logotype_frac=0.9)
    
    #calculate locations from mm to px
    #locations = [0, 0.5, 1.0]
    locations_px = [round(x / ref_ratio + outlet_start) for x in locations] 
    
    #define output arrays
    avg_d_arr = []
    avg_l_arr = []
    phi_arr = []
    stable_d_arr = []
    loc_d = []
    
    #measure jet for each frame
    print('Calculating jet parameters for each frame... (this will take a while)')
    count_frames = np.arange(0,len(sub_jet_frames_arr),1)
    for frame,i in zip(sub_jet_frames_arr,count_frames):
        try:
            if i%50 == 0: 
                sys.stdout.write(f'Progress {np.ceil((i/len(sub_jet_frames_arr))*100)}%         \r')
                sys.stdout.flush()
            data = measureJetFrameNum('', frame, outlet_start, locations_px, left_cut, right_cut)
            if data[0] > 0: #in case if some frame is corrupted, e.g. just white, without a jet
                avg_d_arr.append(data[0])
                avg_l_arr.append(data[1])
                phi_arr.append(data[2])
                stable_d_arr.append(data[10])
                loc_d.append(data[11])
        except:
            print('Error. Frame {}/{} corrupted'.format(i,len(sub_jet_frames_arr)))
            sys.stdout.write(f'Progress {np.ceil((i/len(sub_jet_frames_arr))*100)}%         \r')
            sys.stdout.flush()
    print('Calculating jet parameters for each frame COMPLETED.')
    
    
    #calculate jet parameters in mm and degrees
    avg_d_arr_mm = [i* ref_ratio for i in avg_d_arr]
    avg_l_arr_mm = [i* ref_ratio for i in avg_l_arr]
    stable_d_arr_mm = [i* ref_ratio for i in stable_d_arr]
    avg_jet_d = np.average(avg_d_arr_mm)
    avg_jet_l = np.average(avg_l_arr_mm)
    avg_jet_phi = np.average(phi_arr)
    avg_jet_stable_d = np.average(stable_d_arr_mm)
    d_min = np.min(avg_d_arr_mm)
    d_max = np.max(avg_d_arr_mm)
    l_min = np.min(avg_l_arr_mm)
    l_max = np.max(avg_l_arr_mm)
    d_std = np.std(avg_d_arr_mm)
    l_std = np.std(avg_l_arr_mm)
    stable_d_min = np.min(stable_d_arr_mm)
    stable_d_max = np.max(stable_d_arr_mm)
    
    d_location = []
    loc_name = []
    for i in range(0, len(loc_d), 1):
        dl = []
        for j in range(0, len(loc_d[i]), 1):
            dl.append(ref_ratio * loc_d[i][j][0])
            
            if i == 0:
                loc_name.append(round(ref_ratio * (loc_d[i][j][1] - outlet_start), 4) )
        d_location.append(dl)
        
    #write data
    print('Writing data to .txt file...')
    out_d = ['average jet diameter [mm]',avg_jet_d]
    out_l = ['average jet length [mm]',avg_jet_l]
    out_phi = ['average jet skewness [deg]',avg_jet_phi]
    out_ratio = ['mm to px ratio [mm/px]',ref_ratio]
    out_d_min = ['Minimal d [mm]', d_min]
    out_d_max = ['Maximal d [mm]', d_max]
    out_l_min = ['Minimal l [mm]', l_min]
    out_l_max = ['Maximal l [mm]', l_max]
    out_d_std = ['Standard deviation of d [mm]', d_std]
    out_l_std = ['Standard deviation of l [mm]', l_std]
    out_stable_d = ['average stable jet diameter [mm]',avg_jet_stable_d]
    out_stable_d_min = ['Minimal stable d [mm]', stable_d_min]
    out_stable_d_max = ['Maximal stable d [mm]', stable_d_max]
    fname = video_path[0:-4] + '_AVG_data'+note+'.txt'
    DL.save2DDataToTxt([out_d,out_l,out_phi,out_ratio,out_d_min,out_d_max,out_l_min,out_l_max,out_d_std,out_l_std,out_stable_d,out_stable_d_min,out_stable_d_max], filename=fname, transpose = True)
    
    out_d_frame = ['average jet diameter by frame [mm]'] + avg_d_arr_mm
    out_l_frame = ['average jet length by frame [mm]'] + avg_l_arr_mm
    out_phi_frame = ['average jet skewness by frame [deg]'] + phi_arr
    out_stable_d_frame = ['average stable jet diameter by frame [mm]'] + stable_d_arr_mm
    fname_frame = video_path[0:-4] + '_ALL_data'+note+'.txt'
    DL.save2DDataToTxt([out_d_frame,out_l_frame,out_phi_frame,out_stable_d_frame], filename=fname_frame, transpose = True)
    
    
    out_loc_d_frame = [['jet diameter at location ' + str(x) for x in loc_name]] + d_location
    fname_loc = video_path[0:-4] + '_location_data'+note+'.txt'
    DL.save2DDataToTxt(out_loc_d_frame, filename=fname_loc, transpose = False)
    
    
    
    #extract flow parameters
    head, tail = os.path.split(video_path)
    video_name = tail[:-4]
    G = video_name[1:3]
    L = video_name[4:]
    
    
    #return data
    print(f'Calculation for video {video_path} COMPLETE!')
    print('\n')
    out = (G,L,avg_jet_d, avg_jet_l, avg_jet_phi, ref_ratio,d_min,d_max,l_min,l_max,d_std,l_std,avg_jet_stable_d,stable_d_min,stable_d_max,d_location)
    return out


"---------------------------------------EXPERIMENTAL LIBRARY FOR VIDEO POST-PROCESSING-----------------------------------"

def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def defineStartJetExp(nozzle, background, first_n_lines=100, crop=5, next_nth=5 , save=False):
    '''
    @author: kristof
    
        Defines a start of the jet, which is the first line after the nozzle's orrifice ends.

        Input parameters:
            - nozzle [Image path] "image of the background frame (colored/RGB picture)",
            - background [Image path] "image of the background frame (colored/RGB picture)",
            - first_n_lines [int] "nr. of lines to skip from first line"
            - crop [int] "column numbers to crop an image from left and from right edge"
            - next_nth [int] "number of pixels to check after the last black one pixel at the end of the nozzle's outlet" 
            - save [Bool] "save binarized image. Default is False"
        Return parameters:
            - define start [float] "number of line where jet start, outside the nozzle"
    '''    
    bck_frame = VL.importImage(background)
    ref_frame = VL.importImage(nozzle)

    #substract background from nozzle frame
    a_ = IC.subtract(ref_frame, bck_frame, offset=255)
      
    ref_bin =VL.convertToBinary([a_])
    
    #This block crops an image
    im_a = ref_bin[0]
    im_b = im_a[0:first_n_lines, crop:-crop]
    
    
    #This block finds the last line in colums with black(0) value
    end_value = []
    
    for col in range(0, len(im_b[0]), 1):
        for lin in range(0, len(im_b)-next_nth, 1):
                        
            if im_b[lin][col] == 0:
                next_ = []
                for i in range(0, next_nth, 1):
                    next_.append(im_b[lin+1+i][col])
                
                if all_equal(next_) and next_[0]==255:
                        end_value.append(lin)
    
    #This block finds the most common value(line, where nozzle ends). 
    #If there are more most common lines than one, it picks a maximum one.
    #+1 because jet starts one line below the nozzle's outlet
    define_start = max(multimode(end_value)) + 1 
    
    print("Line of the jet start in px: ", define_start)
    
    
    if save:
        im = Image.fromarray(im_a.astype("uint8")).convert("RGB")
        shape = [(0, define_start), (len(im_a), define_start)]
        im1 = ImageDraw.Draw(im)
        im1.line(shape, fill = (255, 0, 0), width=1)
        im.save("Nozzle.jetStart.jpg")
        
    return (define_start)


def calcRefExp(reference, background, scale, jet_start, skip_last_lines = 20, crop = 5, next_nth = 10, prev_nth = 10, save=False):
    '''
    @author: kristof
    
        Calculates the mm per pixel ratio for CFD Post procesing.

        Input parameters:
            - reference [Image path] "reference image of the background frame (colored/RGB picture)",
            -scale [mm] "reference object dimension"
            - jet_start [int] "line in px where jet starts/nozzle ends. Reference object should be placed with a gap of 10 px under the nozzle's outlet"
            - crop [int] "column numbers to crop an image from left and from right edge"
            - next_nth [int] "number of pixels to check after the last black one pixel at the end of reference object" 
            - prev_nth [int] "number of pixels to check before the first black one pixel at the start of reference object" 
            - save [Bool] "save binarized image. Default is False"
        Return parameters:
            - ratio [float] "mm/px ratio. Multiply this ratio to any given px value to calculate the true value in mm"
            - ref_length [float] diameter of the reference object in px"
    '''
    bck_frame = VL.importImage(background)
    ref_frame = VL.importImage(reference)

    #substract background from nozzle frame
    a_ = IC.subtract(ref_frame, bck_frame, offset=255)
      
    ref_bin =VL.convertToBinary([a_])
    
    #This block crops an image
    im_a = ref_bin[0]
    im_b = im_a[jet_start+10:-skip_last_lines, crop:-crop]
    
    
    #This block finds the start and end line of the reference object
    start_value = []
    end_value = []
    
    for col in range(0, len(im_b[0]), 1):
        for lin in range(0, len(im_b)-next_nth, 1):
                        
            if im_b[lin][col] == 0:
                next_ = []
                previous_ = []
                
                for i in range(0, next_nth, 1):
                    next_.append(im_b[lin+1+i][col])
                
                if all_equal(next_) and next_[0]==255:
                        end_value.append(lin)
                
                for j in range(0, prev_nth, 1):
                    previous_.append(im_b[lin-1-j][col])
                
                if all_equal(previous_) and previous_[0]==255:
                        start_value.append(lin)                               
    
    #This block finds the most common value(line, where nozzle ends). 
    #If there are more most common lines than one, it picks a maximum one.
    ref_start = max(multimode(start_value)) 
    ref_end = max(multimode(end_value)) 
    
    ref_length = ref_end - ref_start + 1
    
    print("Reference length in px: ", ref_length)
    
    ratio = scale / ref_length
    
    print("Ratio mm per px: ", ratio)
    
    if save:
        im = Image.fromarray(im_a.astype("uint8")).convert("RGB")
        shape1 = [(0, ref_start + jet_start + 10), (len(im_a), ref_start + jet_start + 10)]
        shape2 = [(0, ref_end + jet_start + 10), (len(im_a), ref_end + jet_start + 10)]
        shape = [(0, jet_start), (len(im_a), jet_start)]
        im1 = ImageDraw.Draw(im)
        
        im1.line(shape1, fill = (255, 0, 0), width=1)
        im1.line(shape2, fill = (255, 0, 0), width=1)
        im1.line(shape, fill = (255, 0, 0), width=1)

        im.save("Nozzle.Reference.jpg")

    return (ratio, ref_length)



def measureJetFrameExp(path, frame, outlet_start, locations_px, left_cut = 20, right_cut = 20, next_nth = 5, prev_nth = 5, saveImage = False, outputf = 'measured_lines/'):
    '''
    @author: kristof
    
        This function extracts jet parameters from a .jpg frame of a jet.

        Input parameters:
            - path [string] "path of the frame",
            - frame [PIL.Image] "Image object of the frame, binary PIL.Image",
            - outlet_start [line number] "line where jet starts, use defineStartJetExp function ", 
            - locations_px [list(float)] "list of locations (in px) at which the jet diameters wanted to be checked"
            - left_cut [column nr.] "column number of the frame to cut the image, e.g. logotypes, scales, axis etc.",
            - right_cut [column nr.] "column number of the frame to cut the image, e.g. logotypes, scales, axis etc.",
            - next_nth [int] "number of pixels to check after the last black one pixel at the end of reference object", 
            - prev_nth [int] "number of pixels to check before the first black one pixel at the start of reference object", 
            - saveImage [bool] "True/False save image with calculated parameters",
            - outputf [string] "path to output folder"
        Return parameters:
            - d0 [float] "average diameter of the jet with correction for skewness",
            - l0 [float] "average length of the jet with correction for skewness",
            - np.degrees(phi) [float] "skewness angle in degrees",
            - threshold_d [float] "threshold diameter calculated from lines 10:20 of the jet",
            - jet_line_start [int] "frame line (from [0,height]) where jet starts",
            - jet_line_end [int] "frame line (from [0,height]) where jet ends",
            - jet_line_end_stable [int] "frame line (from [0,height]) where stable jet ends",
            - jet_arr [array((array(int),int))] "array of image line pixels (values from (0,255)) and frame line height ",
            - jet_d_arr [array((int,int,int))] "array of tuples where the [0] stores the number of pixels across the jet, [1] frame line height and [2] number of pixels to the beginning of the jet from the left",
            - jet_d_arr_stable [array((int,int,int))] "stores the same values as jet_d_arr but only to the end of the stable jet defined by breakRatio",
            - stable_d0 [float] "diameter of the stable jet with correction for skewness"
    '''
    
    #this block defines start of the jet, which is nozzle's outlet line in the image      
    #outlet_start = defineStartJetNum(reference)
    
    
    #cuting image, e.g. logotypes, scales, axis, etc.  
    img = np.asarray(frame[:, left_cut:-right_cut])
    
    width = len(img[0])
    jet_line_start = 0 
    jet_line_end = 0
    
    #this block gets an array of line pixel values over the whole jet without droplets
    jet_arr = [] #include frame lines of jet
    count = -1
    check_jet_start = False #True when jet startes
    for line in img: #loop through the frame pixel lines and extract lines that include the jet
        count += 1
        if np.average(line)  != 255: check_jet_start = True
        if np.average(line) == 255 and check_jet_start: break
        #if np.average(line) > 255*0.99: continue
        jet_arr.append((line,count))
    
    #this block takes into account the start of the jet at the nozzle's outlet
    jet_arr = jet_arr[outlet_start:]
    
    
    #next_nth = 5
    #prev_nth = 5
    
      
    #this block calculates the average diameter of the first n lines of the jet
    threshold_d_arr = [] #list of first n line jet diameters starting from mth line (to remove first unstable lines)
    for line in jet_arr[10:20]: #10:20
        line = line[0]
        _jet = False #True when jet starts
        
    #line_test = jet_arr[10]
    #line = line_test[0]
    
        start_value = []
        end_value = []
        
        for col in range(0, len(line)-next_nth, 1):
                    
            if line[col] == 0:
                next_ = []
                previous_ = []
                
                for i in range(0, next_nth, 1):
                    next_.append(line[col+1+i])
                
                if all_equal(next_) and next_[0]==255:
                        end_value.append(col)
                
                for j in range(0, prev_nth, 1):
                    previous_.append(line[col-1-j])
                
                if all_equal(previous_) and previous_[0]==255:
                        start_value.append(col)                               
        
        #This block excludes possible dots, which are consequence of camera pixel error and are visible on the background, and could be recognized as a jet start/end
        """
        if len(start_value) > 2.1:
            start_value_filter = []
            end_value_filter = []
            diference = []
            for i in range(0, len(start_value), 1):
                _dif = end_value[i] - start_value[i]
                diference.append(_dif)
            
            for i in range(0, len(diference)-1, 1):    
                if diference[i+1]/diference[i] < 1.5 and diference[i+1]/diference[i] > 0.5:
                    start_value_filter.append(start_value[i]) 
                    start_value_filter.append(start_value[i+1])    
                    end_value_filter.append(end_value[i]) 
                    end_value_filter.append(end_value[i+1])  
        
                start_value = start_value_filter
                end_value = end_value_filter
        """
        #This block finds the most common value(line, where nozzle ends). 
        #If there are more most common lines than one, it picks a maximum one.
        ref_start = min(start_value) 
        ref_end = max(end_value)
        
        ref_length = ref_end - ref_start + 1
        
        threshold_d_arr.append(ref_length)
        
        
        
        
    """    
        count_px = 0 #counts the number of pixel across the jet diameter
        for i in range(0,width):
            if np.all(line[i:i+5] < 10): _jet = True
            if _jet and line[i] > 10: break
            if _jet: count_px += 1
        if count_px < 50: threshold_d_arr.append(count_px) #if the nozzle in the background was to high and there is a thick line at the top
    """
    threshold_d = np.average(threshold_d_arr) #average diameter of first 10 lines    
    
    #this block calculates the diameter of the jet.
    jet_d_arr = [] #array of jet diameters
    
    jet_arr_all = jet_arr
    
    jet_arr = jet_arr[2:] #skips the first two lines of the jet, because of poble interaction wih nozzle's outlet
    
    
    for line in jet_arr:
        line_num = line[1]
        line = line[0]
        #_jet = False #True when jet starts
        
        #count_px
        start_value = []
        end_value = []
        
        for col in range(0, len(line)-next_nth, 1):
                    
            if line[col] == 0:
                next_ = []
                previous_ = []
                
                for i in range(0, next_nth, 1):
                    next_.append(line[col+1+i])
                
                if all_equal(next_) and next_[0]==255:
                        end_value.append(col)
                
                for j in range(0, prev_nth, 1):
                    previous_.append(line[col-1-j])
                
                if all_equal(previous_) and previous_[0]==255:
                        start_value.append(col)                               
        
        
        #This block excludes possible dots, which are consequence of camera pixel error and are visible on the background, and could be recognized as a jet start/end
        """
        if len(start_value) > 2:
            start_value_filter = []
            end_value_filter = []
            diference = []
            for i in range(0, len(start_value), 1):
                _dif = end_value[i] - start_value[i]
                diference.append(_dif)
            
            for i in range(0, len(diference)-1, 1):    
                if diference[i+1]/diference[i] < 1.5 and diference[i+1]/diference[i] > 0.5:
                    start_value_filter.append(start_value[i]) 
                    start_value_filter.append(start_value[i+1])    
                    end_value_filter.append(end_value[i]) 
                    end_value_filter.append(end_value[i+1])  
        
                start_value = start_value_filter
                end_value = end_value_filter
        """
        
        #This block finds the most common value(line, where nozzle ends). 
        #If there are more most common lines than one, it picks a maximum one.
        ref_start = min(start_value)
        ref_end = max(end_value)
        
        ref_length = ref_end - ref_start + 1    
           
        count_px = ref_length #counts the number of pixel across the jet diameter
        
        count_px_left = ref_start - 1 #counts the number of pixel before jet starts
        
        """
        for i in range(0,width):
            if np.all(line[i:i+5] < 10): _jet = True
            if _jet and line[i] > 10: break
            if _jet: count_px += 1
            else: count_px_left += 1
        """
        jet_d_arr.append((count_px, line_num, count_px_left))
    
        
    #This block calculates the diameter of the jet at desired locations
    jet_d_loc = []
    for i in range(0, len(locations_px), 1):
        for j in range(0, len(jet_d_arr), 1):
            if locations_px[i] == jet_d_arr[j][1]:
                jet_d_loc.append(jet_d_arr[j][0:-1])
    
        if locations_px[i] > jet_d_arr[-1][1]:
            jet_d_loc.append((0, locations_px[i]))
    
    
    #remove first unstable values unless the frame has no nozzle trace. check for trace with standard deviation of the first 15 jet diameters (if there is none it should be under 1)
    first_array = [i[0] for i in jet_d_arr[0:15]]
    first_array_std = np.std(first_array)
    
    if first_array_std > 3:
        _pos = 0
        _koef = 5 #koeficient, 5
        for i in range(0,20):
            if np.abs(jet_d_arr[20-i][0] - threshold_d) > _koef: _pos = i; break
        jet_d_arr = jet_d_arr[21-_pos:]
        jet_line_start += jet_d_arr[0][1]
    
    
    #set where jet structure ends
    jet_line_end = jet_d_arr[-1][1]
    
    
    #check end of stable jet by variance of the diameter
    jet_d_arr_stable = jet_d_arr[0:10]
    
    jet_d_arr_stable = jet_d_arr[0:10]
    _len = len(jet_d_arr)
    for i in range(10,_len):
        var_arr = jet_d_arr[i-4:i+1] #i-5
        var_arr = [i[0] for i in var_arr]
        var = np.var(var_arr)
        #print(i,var)
        if var>1 : break
        jet_d_arr_stable.append(jet_d_arr[i])
    
    jet_line_end_stable = jet_d_arr_stable[-1][1]   
    
    #calculate average diameter of the stable jet
    stable_d_arr = []
    for values in jet_d_arr_stable:
        stable_d_arr.append(values[0])
    avg_stable_d_px = np.average(stable_d_arr)
    
    
    #calculate average jet diameter and jet length in case jet is straight
    jet_d = [i[0] for i in jet_d_arr]
    jet_avg_d_px = np.average(jet_d)
    jet_l_px = jet_d_arr[-1][1] - jet_arr_all[0][1]
    
    
    #calculate jet skewness
    L1 = (jet_d_arr[0][2], jet_d_arr[0][1])
    R1 = (jet_d_arr[0][2] + jet_d_arr[0][0], jet_d_arr[0][1])
    L2 = (jet_d_arr_stable[-1][2], jet_d_arr_stable[-1][1])
    R2 = (jet_d_arr_stable[-1][2] + jet_d_arr_stable[-1][0], jet_d_arr_stable[-1][1])
    M1 = (int(np.ceil(0.5*(L1[0] + R1[0]))), jet_d_arr_stable[0][1])
    M2 = (int(np.ceil(0.5*(L2[0] + R2[0]))), jet_d_arr_stable[-1][1])
    L3 = (jet_d_arr[-1][2]-10, jet_d_arr[-1][1])
    R3 = (jet_d_arr[-1][2] + jet_d_arr[-1][0]+10, jet_d_arr[-1][1])
    STABLE1 = (M2[0]-10,M2[1])
    STABLE2 = (M2[0]+10,M2[1])
    
    
    #calculate true average diameter and length
    sk = M2[0] - M1[0]
    l = jet_l_px + 1
    d = jet_avg_d_px
    phi = np.arctan(sk/l)
    l0 = l/np.cos(phi)
    d0 = d*np.cos(phi)
    stable_d0 = avg_stable_d_px*np.cos(phi) 
    
    #make image
    if saveImage:
        _,file_name = os.path.split(path)
        Path(outputf).mkdir(parents=True, exist_ok=True)
        PL.drawLines(path,output = outputf+file_name,points = [(L1[0], L1[1], R1[0], R1[1]), (L3[0], L3[1], R3[0], R3[1]), (M1[0], M1[1], M2[0], M2[1]), (STABLE1[0], STABLE1[1], STABLE2[0], STABLE2[1])],color = (255,0,0),thickness = 1) #outputs .jpg with marked start od end of the stable jet
    
    #this block takes into account the start of jet at the nozzle's outlet line
    jet_line_start = outlet_start + jet_line_start    
    
    #return arguments
    return (d0, l0, np.degrees(phi), threshold_d, jet_line_start, jet_line_end, jet_line_end_stable, jet_arr, jet_d_arr, jet_d_arr_stable,stable_d0, jet_d_loc)
    



def measureJetVideoExp(video_path, bcg_frame_path, jet_start, mm_to_px_ratio, locations, start_frame = 0, end_frame = -1, step=1, left_cut = 20, right_cut = 20, next_nth = 5, prev_nth = 5, note=''):
    """
    @author: kristof
    
        Measures jet parameters over a whole video.
    
        Input parameters:
            - video_path [string] "path to the video",
            - bcg_frame_path [string] "path to the background frame. You have to use getFrames function outside this loop",
            - jet_start [int] "number of line, where jet starts. Use an output of the defineStartJetExp function."
            - mm_to_px_ratio [float] "ratio mm to px. Use an output of the calcRefExp function "
            - locations [list(float)] "list of locations (in mm or other working unit) at which the jet diameters wanted to be checked"
            - start_frame [int] "select starting frame",
            - end_frame [int] "select ending frame",
            - step [int] "step to extract each n frame",
            - left_cut [column nr.] "column number of the frame to cut the image, e.g. logotypes, scales, axis etc.",
            - right_cut [column nr.] "column number of the frame to cut the image, e.g. logotypes, scales, axis etc.",
            - next_nth [int] "number of pixels to check after the last black one pixel at the end of jet object", 
            - prev_nth [int] "number of pixels to check before the first black one pixel at the start of jet object",
            - note [string] "add note to file name for AVG and ALL data"
        Return parameters:
            - G [int] "gas volume flow in l/h",
            - L [int] "liquid volume flow in ul/min",
            - avg_jet_d [float] "average jet diameter in mm across the frames",
            - avg_jet_l [float] "average jet length in mm across the frames",
            - avg_jet_phi [float] "average jet skewness in degrees across the frames",
            - ref_ratio [float] "mm/px ratio"
            - d_min [float] "minimal jet diameter in mm",
            - d_max [float] "maximal jet diameter in mm",
            - l_min [float] "minimal jet length in mm",
            - l_max [float] "maximal jet length in mm",
            - d_std [float] "diameter standard deviation in mm",
            - l_std [float] "length standard deviation in mm",
            - avg_jet_stable_d [float] "average stable jet diameter in mm across the frames in mm",
            - stable_d_min [float] "minimal stable jet diameter in mm",
            - stable_d_max [float] "maximal stable jet diameter in mm"
    """
    
    #extract frames in temp folders, import them as PIL.Image objects and delete temp files
    print('Starting calculation for '+video_path)
    print('Preparing frames...')
    VL.getFrames(video_path, end_frame=end_frame, start_frame=start_frame, step=step, BW = True, save_path='_temp_video_frames')
    #VL.getFrames(bcg_path, BW = True, save_path='_temp_bcg_frames')
    #VL.getFrames(ref_path, BW = True, save_path='_temp_ref_frames')
    
    video_frames_path = DL.getFiles(path = '_temp_video_frames/', extension = '.jpg')
    #bcg_frame_path = DL.getFiles(path = '_temp_bcg_frames/', extension = '.jpg')
    #ref_frame_path = ref_path
    
    video_frames_arr = []
    bcg_frame = VL.importImage(bcg_frame_path)
    #bcg_frame = VL.importImage(bcg_frame_path[0])
    #ref_frame = VL.importImage(ref_frame_path[0])
    
    for path in video_frames_path:
        video_frames_arr.append(VL.importImage(path))
    
    print('Deleting temporary files...')
    shutil.rmtree('_temp_video_frames/')
    #shutil.rmtree('_temp_bcg_frames/')
    #shutil.rmtree('_temp_ref_frames/')
    print('Temporary files deleted.')
    
    
    #substract background from jet and binarize frames
    print('Substracting background and binarizing frames...')
    sub_jet_frames_arr = []
    for i in video_frames_arr:
        _ = IC.subtract(i, bcg_frame, offset=255)
        _bin = VL.convertToBinary([_])
        sub_jet_frames_arr.append(_bin[0])
    
    #defines nozzle's outlet line on image
    print("Calculating nozzle's outlet line...")
    outlet_start =jet_start
    
    #calculate mm/px ratio
    print('Calculating mm/px ratio...')
    #_ref_sub = IC.subtract(ref_frame, bcg_frame, offset=255)
    #ref_bin_frame = VL.convertToBinary([_ref_sub])
    ref_ratio = mm_to_px_ratio
    
    #calculate locations from mm to px
    #locations = [0, 0.5, 1.0] #example
    locations_px = [round(x / ref_ratio + outlet_start) for x in locations] 
    
    #define output arrays
    avg_d_arr = []
    avg_l_arr = []
    phi_arr = []
    stable_d_arr = []
    loc_d = []
    
    #measure jet for each frame
    print('Calculating jet parameters for each frame... (this will take a while)')
    count_frames = np.arange(0,len(sub_jet_frames_arr),1)
    for frame,i in zip(sub_jet_frames_arr,count_frames):
        try:
            if i%50 == 0: 
                sys.stdout.write(f'Progress {np.ceil((i/len(sub_jet_frames_arr))*100)}%         \r')
                sys.stdout.flush()
            data = measureJetFrameExp('', frame, outlet_start, locations_px, left_cut, right_cut, next_nth, prev_nth, saveImage = False, outputf = 'measured_lines/')
            if data[0] > 0: #in case if some frame is corrupted, e.g. just white, without a jet
                avg_d_arr.append(data[0])
                avg_l_arr.append(data[1])
                phi_arr.append(data[2])
                stable_d_arr.append(data[10])
                loc_d.append(data[11])
        except:
            print('Error. Frame {}/{} corrupted'.format(i,len(sub_jet_frames_arr)))
            sys.stdout.write(f'Progress {np.ceil((i/len(sub_jet_frames_arr))*100)}%         \r')
            sys.stdout.flush()
    print('Calculating jet parameters for each frame COMPLETED.')
    
    
    #calculate jet parameters in mm and degrees
    avg_d_arr_mm = [i* ref_ratio for i in avg_d_arr]
    avg_l_arr_mm = [i* ref_ratio for i in avg_l_arr]
    stable_d_arr_mm = [i* ref_ratio for i in stable_d_arr]
    avg_jet_d = np.average(avg_d_arr_mm)
    avg_jet_l = np.average(avg_l_arr_mm)
    avg_jet_phi = np.average(phi_arr)
    avg_jet_stable_d = np.average(stable_d_arr_mm)
    d_min = np.min(avg_d_arr_mm)
    d_max = np.max(avg_d_arr_mm)
    l_min = np.min(avg_l_arr_mm)
    l_max = np.max(avg_l_arr_mm)
    d_std = np.std(avg_d_arr_mm)
    l_std = np.std(avg_l_arr_mm)
    stable_d_min = np.min(stable_d_arr_mm)
    stable_d_max = np.max(stable_d_arr_mm)
    
    d_location = []
    loc_name = []
    for i in range(0, len(loc_d), 1):
        dl = []
        for j in range(0, len(loc_d[i]), 1):
            dl.append(ref_ratio * loc_d[i][j][0])
            
            if i == 0:
                loc_name.append(round(ref_ratio * (loc_d[i][j][1] - outlet_start), 4) )
        d_location.append(dl)
    
    #write data
    print('Writing data to .txt file...')
    out_d = ['d_avg [mm]', round(avg_jet_d, 8)]
    out_l = ['l_avg [mm]', round(avg_jet_l, 8)]
    out_phi = ['sk_avg [deg]', round(avg_jet_phi, 8)]
    out_ratio = ['ratio [mm/px]', round(ref_ratio, 8)]
    out_d_min = ['d_min [mm]', round(d_min, 8)]
    out_d_max = ['d_max [mm]', round(d_max, 8)]
    out_l_min = ['l_min [mm]', round(l_min, 8)]
    out_l_max = ['l_max [mm]', round(l_max, 8)]
    out_d_std = ['d_std [mm]', round(d_std, 8)]
    out_l_std = ['l_std [mm]', round(l_std, 8)]
    out_stable_d = ['d_s_avg [mm]', round(avg_jet_stable_d, 8)]
    out_stable_d_min = ['d_s_min [mm]', round(stable_d_min, 8)]
    out_stable_d_max = ['d_s_max [mm]', round(stable_d_max, 8)]
    fname = video_path[0:-4] + '_AVG_data'+note+'.txt'
    DL.save2DDataToTxt([out_d,out_l,out_phi,out_ratio,out_d_min,out_d_max,out_l_min,out_l_max,out_d_std,out_l_std,out_stable_d,out_stable_d_min,out_stable_d_max], filename=fname, transpose = True)
    
    out_d_frame = ['d_avg by frame [mm]'] + avg_d_arr_mm
    out_l_frame = ['l_avg by frame [mm]'] + avg_l_arr_mm
    out_phi_frame = ['sk_avg by frame [deg]'] + phi_arr
    out_stable_d_frame = ['d_s_avg by frame [mm]'] + stable_d_arr_mm
    fname_frame = video_path[0:-4] + '_ALL_data'+note+'.txt'
    DL.save2DDataToTxt([out_d_frame,out_l_frame,out_phi_frame,out_stable_d_frame], filename=fname_frame, transpose = True)
    
    out_loc_d_frame = [['jet diameter at location ' + str(x) for x in loc_name]] + d_location
    fname_loc = video_path[0:-4] + '_location_data'+note+'.txt'
    DL.save2DDataToTxt(out_loc_d_frame, filename=fname_loc, transpose = False)
    
    
    #extract flow parameters
    head, tail = os.path.split(video_path)
    video_name = tail[:-4]
    G = video_name[1:3]
    L = video_name[4:]
    
    
    #return data
    print(f'Calculation for video {video_path} COMPLETE!')
    print('\n')
    out = (G,L,avg_jet_d, avg_jet_l, avg_jet_phi, ref_ratio,d_min,d_max,l_min,l_max,d_std,l_std,avg_jet_stable_d,stable_d_min,stable_d_max, d_location)
    return out
    
