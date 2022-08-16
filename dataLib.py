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

File 'dataLib.py' contains functions for data import and data export.
'''
import os
import ntpath

def save2DDataToTxt(input, filename='2DData.txt', transpose = False):
    '''
        Saves array of data into .txt file in tabular form ready to be imported into .csv file.

        Input parameters:
            - input [array(array(float), array(float), ..., array(float))] "2D array of values eg. [[i1,i2,i3], [j1,j2,j3], [k1,k2,k3]]"
            - filename [string] "path of the generated .txt file. Must include the extension .txt"
            - transpose [bool] "True/False transpose the matrix eg. [[i1,i2,i3], [j1,j2,j3], [k1,k2,k3]] -> [[i1,j1,k1], [i2,j2,k2], [i3,j3,k3]]"
    '''

    if transpose: input = map(list, zip(*input))
    _ = open(filename,'w')
    for i in input:
        l = len(i)
        count = 1
        for j in i:
            _.write(str(j))
            if count != l: _.write('\t')
            count += 1
        _.write('\n')
    _.close()

    return 'Created 2D file '+filename

def save1DDataToTxt(input, filename='file.txt'):
    '''
        Saves array of data into .txt file in tabular form ready to be imported into .csv file.

        Input parameters:
            - input [array(float)] "1D array of values eg. [i1,i2,i3]"
            - filename [string] "path of the generated .txt file. Must include the extension .txt"
    '''

    _ = open(filename,'w')
    for i in input:
        _.write(str(i) + '\n')
    _.close()

    return 'Created 1D file '+filename

def import2DDataFromTxt(filename, skip = 0, transpose = False):
    '''
        Imports array of data from .txt file.

        Input parameters:
            - filename [string] "path of the .txt file. Must include the extension .txt"
            - skip [int] "number of lines in the file to skip (eg. file has headers)"
            - transpose [bool] "True/False transpose the matrix eg. [[i1,j1,k1], [i2,j2,k2], [i3,j3,k3]] -> [[i1,i2,i3], [j1,j2,j3], [k1,k2,k3]]"
        Return parameters:
            - data [array(array(float), array(float), ..., array(float))] "2D array of values eg. [[i1,i2,i3], [j1,j2,j3], [k1,k2,k3]]"
    '''

    with open(filename) as f:
        content = f.readlines()
        content = content[skip::]
        data = []
        for i in content:
            arr = []
            _ = i.split('\t')
            for j in _:
                try:
                    arr.append(float(j))
                except:
                    arr.append(j.rstrip())
            data.append(tuple(arr))
    if transpose: data = list(map(list, zip(*data)))
    return data

def import1DDataFromTxt(filename, skip = 0):
    '''
        Imports array of data from .txt file.

        Input parameters:
            - filename [string] "path of the .txt file. Must include the extension .txt"
            - skip [int] "number of lines in the file to skip (eg. file has headers)"
        Return parameters:
            - data [array(float)] "1D array of values eg. [i1,i2,i3]"
    '''

    with open(filename) as f:
        content = f.readlines()
        content = content[skip::]
        data = []
        for i in content:
            data.append(float(i))
    return data

def getFiles(path = '', extension = ''):
    '''
        Returns the names of all files in the given folder with the specified extension.

        Input parameters:
            - path [string] "path of the folder. Must include / at the end eg. 'folder/subfolder/'. Leave empty if you are searching in the current working directory"
            - extension [string] "extension of the files eg. .txt, .mp4, .AVI, .avi"
        Return parameters:
            - output [array(string)] "array of paths"
    '''

    output = []
    mypath = os.getcwd()
    if path != '': mypath += '/' + path
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    for i in onlyfiles:
        if i.find(extension) != -1:
            output.append(path+i)
    if len(output) == 0: print('No file with extension "'+extension+'" was found!')
    return output

def extractFileNameFromPath(fpath):
    '''
        Extracts file name from path.

        Input parameters:
            - fpath [string] "path of the specified file"
    '''

    head, tail = ntpath.split(fpath)
    return tail or ntpath.basename(head)

