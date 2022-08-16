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

File 'plotLib.py' contains functions for plotting data and drawing on stil images.
'''
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import cv2

#TODO : REORGANIZE AND SIMPLIFY WITH *args **kwargs

def plotLine(x,value,label='',title = '', xlabel = '', ylabel = '', xlim = (0,0), ylim = (0,0), showPoints = True, grid = False, opacity = 1, save = False, filename = 'plot.pdf', showPlot = False):
    '''
        Draw a line plot.

        Input parameters:
            - x [array(float)] "x values",
            - value [array(float)] "y values",
            - label [string] "label for the plotted line.Supports LaTeX notation; to use it type between $ symbol eg. $\dot{V}_g = 0.887 m^3s^{-1}$",
            - title [string] "Title of the graph. Supports LaTeX notation; to use it type between $ symbol eg. $\dot{V}_g = 0.887 m^3s^{-1}$",
            - xlabel [string] "X axis label of the graph. Supports LaTeX notation; to use it type between $ symbol eg. $\dot{V}_g = 0.887 m^3s^{-1}$",
            - ylabel [string] "Y axis label of the graph. Supports LaTeX notation; to use it type between $ symbol eg. $\dot{V}_g = 0.887 m^3s^{-1}$",
            - xlim [tuple(int,int)] "X axis limit. Leave empty to include all data. eg. (0,100)",
            - ylim [tuple(int,int)] "Y axis limit. Leave empty to include all data. eg. (0,100)",
            - grid [bool] "True/False draw a grid",
            - opacity [int] "Set the opacity of the line. Value between (0,1)",
            - save [bool] "True/False save plot as image",
            - filename [string] "name of he saved file. Extension can be set to .pdf, .jpg, .png ...",
            - showPlot [bool] "True/False show plot in external window"
    '''

    plt.clf()
    plt.plot(x,value,'b',label = label, alpha = opacity)
    if showPoints: plt.plot(x,value,'r^')
    if xlabel != '': plt.xlabel(xlabel)
    if ylabel != '': plt.ylabel(ylabel)
    if xlim[0] != 0 or xlim[1] != 0 : plt.xlim(xlim)
    if ylim[0] != 0 or ylim[1] != 0 : plt.ylim(ylim)
    if title != '': plt.suptitle(title)
    if label != '': plt.legend()
    if grid: plt.grid()
    if save: plt.savefig(filename)
    if showPlot : plt.show()

def plotLines(values, labels, title = '', xlabel = '', ylabel = '', xlim = (0,0), ylim = (0,0), showPoints = True, grid = False, opacity = 1, save = False, filename = 'plot.pdf', showPlot = False):
    '''
        Draw a multiple line plot.

        Input parameters:
            - values [array( array(array(float), array(float), string), array(array(float), array(float), string), ..., array(array(float), array(float), string) )] "array of arrays where each subarray represents its own line plot. The subarray is structured like so [x, values, label] where x is array(float), values is array(float) and label is string which defines line type and color. See examples"
            - labels [array(string)] "Array that contains the labels for each line (presented in the legend)"
            - title [string] "Title of the graph. Supports LaTeX notation; to use it type between $ symbol eg. $\dot{V}_g = 0.887 m^3s^{-1}$",
            - xlabel [string] "X axis label of the graph. Supports LaTeX notation; to use it type between $ symbol eg. $\dot{V}_g = 0.887 m^3s^{-1}$",
            - ylabel [string] "Y axis label of the graph. Supports LaTeX notation; to use it type between $ symbol eg. $\dot{V}_g = 0.887 m^3s^{-1}$",
            - xlim [tuple(int,int)] "X axis limit. Leave empty to include all data. eg. (0,100)",
            - ylim [tuple(int,int)] "Y axis limit. Leave empty to include all data. eg. (0,100)",
            - showPoints [bool] "True/False show points and lines"
            - grid [bool] "True/False draw a grid",
            - opacity [int] "Set the opacity of the line. Value between (0,1)",
            - save [bool] "True/False save plot as image",
            - filename [string] "name of he saved file. Extension can be set to .pdf, .jpg, .png ...",
            - showPlot [bool] "True/False show plot in external window"
    '''

    count = 0
    plt.clf()
    for i in values:
        name = labels[count]
        plt.plot(i[0],i[1],i[2],alpha = opacity,label=labels[count])
        if showPoints: plt.plot(i[0],i[1],'k^')
        count += 1
    plt.legend()
    if xlabel != '': plt.xlabel(xlabel)
    if ylabel != '': plt.ylabel(ylabel)
    if xlim[0] != 0 or xlim[1] != 0 : plt.xlim(xlim)
    if ylim[0] != 0 or ylim[1] != 0 : plt.ylim(ylim)
    if title != '': plt.suptitle(title)
    if grid: plt.grid()
    if save: plt.savefig(filename)
    if showPlot : plt.show()

def plotContour(x, y, value, title = '', xlabel = '', ylabel = '', colorbarlabel = '', xlim = (0,0), ylim = (0,0), showPoints = True, showValues = True, grid = False, phi = 2, save = False, filename = 'plot.pdf', showPlot = False):
    '''
        Draw a contour plot.

        Input parameters:
            - x [array(float)] "array of x coordinates",
            - y [array(float)] "array of y coordinates",
            - value [array(float)] "array of values",
            - title [string] "Title of the graph. Supports LaTeX notation; to use it type between $ symbol eg. $\dot{V}_g = 0.887 m^3s^{-1}$",
            - xlabel [string] "X axis label of the graph. Supports LaTeX notation; to use it type between $ symbol eg. $\dot{V}_g = 0.887 m^3s^{-1}$",
            - ylabel [string] "Y axis label of the graph. Supports LaTeX notation; to use it type between $ symbol eg. $\dot{V}_g = 0.887 m^3s^{-1}$",
            - colorbarlabel [string] "colorbar label of the graph. Supports LaTeX notation; to use it type between $ symbol eg. $\dot{V}_g = 0.887 m^3s^{-1}$",
            - xlim [tuple(int,int)] "X axis limit. Leave empty to include all data. eg. (0,100)",
            - ylim [tuple(int,int)] "Y axis limit. Leave empty to include all data. eg. (0,100)",
            - showPoints [bool] "True/False show points and lines"
            - showValues [bool] "True/False show value nex to the point"
            - grid [bool] "True/False draw a grid",
            - phi [int] "contour multiplier"
            - save [bool] "True/False save plot as image",
            - filename [string] "name of he saved file. Extension can be set to .pdf, .jpg, .png ...",
            - showPlot [bool] "True/False show plot in external window"
    '''

    conto = len(Counter(value).keys())*phi
    plt.tricontour(x, y, value, conto, linewidths=0.5, colors='k')
    plt.tricontourf(x, y, value, conto, cmap = 'RdBu_r')
    plt.colorbar(label=colorbarlabel,pad=0.1)
    if showPoints: plt.plot(x,y,'kx',alpha=1)
    if showValues:
        for i,j,k in zip(x,y,value):
            plt.text(i+0.2,j+0.2,k,rotation=0)
    if xlabel != '': plt.xlabel(xlabel)
    if ylabel != '': plt.ylabel(ylabel)
    if xlim[0] != 0 or xlim[1] != 0 : plt.xlim(xlim)
    if ylim[0] != 0 or ylim[1] != 0 : plt.ylim(ylim)
    if title != '': plt.suptitle(title)
    if grid: plt.grid()
    if save: plt.savefig(filename)
    if showPlot : plt.show()

def drawLine(path,output = 'output.jpg',point1 = (0,0),point2 = (0,0),color = (255,0,0),thickness = 1):
    '''
        Draws a custom line over an image.

        Input parameters.
            - path [string] "path of the image file",
            - output [string] "path of the output image. Must contain .jpg extension"
            - point1 [tuple(int,int)] "coordinates of the first point",
            - point2 [tuple(int,int)] "coordinates of the second point",
            - color [tuple(int,int,int)] "RGB value. Integers must be between (0,255)",
            - thickness [int] "thickness of the line"
    '''

    image = cv2.imread(path)
    image = cv2.line(image, point1, point2, (color[2],color[1],color[0]), thickness)
    cv2.imwrite(output,image)

def drawLines(path,output = 'output.jpg',points = [],color = (255,0,0),thickness = 1):
    '''
        Draws custom lines over an image.

        Input parameters.
            - path [string] "path of the image file",
            - output [string] "path of the output image. Must contain .jpg extension"
            - points [array( array(int,int,int,int), ..., array(int,int,int,int) )] "Array of subarrays, where each subarray defines a line. Form of subarray is [point1_x, point1_y, point2_x, point2_y,]",
            - color [tuple(int,int,int)] "RGB value. Integers must be between (0,255)",
            - thickness [int] "thickness of the line"
    '''

    image = cv2.imread(path)
    for i in points:
        image = cv2.line(image, (i[0],i[1]), (i[2],i[3]), (color[2],color[1],color[0]), thickness)
    cv2.imwrite(output,image)

def drawMap(data, labels, title = '', xlabel = '', ylabel = '', xlim = (0,0), ylim = (0,0), grid = False, save = False, filename = 'plot.pdf', showPlot = False):
    '''
        Draws a map of points.

        Input parameters:
            - data [array( array( array(array(int,int), array(int,int), ...), string ), array( array(array(int,int), array(int,int), ...), string ), ... )] "array of datasets where each dataset consists of two parts. The second part is the point type eg. 'r^' for red triangles. The first part is an array of points in format (x,y) eg. [(x1,y1), (x2,y2), ..., (xn,yn)]",
            - labels [array(string)] "labels for each set of points. Supports LaTeX notation; to use it type between $ symbol eg. $\dot{V}_g = 0.887 m^3s^{-1}$",
            - title [string] "Title of the graph. Supports LaTeX notation; to use it type between $ symbol eg. $\dot{V}_g = 0.887 m^3s^{-1}$",
            - xlabel [string] "X axis label of the graph. Supports LaTeX notation; to use it type between $ symbol eg. $\dot{V}_g = 0.887 m^3s^{-1}$",
            - ylabel [string] "Y axis label of the graph. Supports LaTeX notation; to use it type between $ symbol eg. $\dot{V}_g = 0.887 m^3s^{-1}$",
            - xlim [tuple(int,int)] "X axis limit. Leave empty to include all data. eg. (0,100)",
            - ylim [tuple(int,int)] "Y axis limit. Leave empty to include all data. eg. (0,100)",
            - grid [bool] "True/False draw a grid",
            - save [bool] "True/False save plot as image",
            - filename [string] "name of he saved file. Extension can be set to .pdf, .jpg, .png ...",
            - showPlot [bool] "True/False show plot in external window"
    '''

    for dataset, label in zip(data,labels):
        x = [i[0] for i in dataset[0]]
        y = [i[1] for i in dataset[0]]
        plt.plot(x, y, dataset[1], label = label)
    plt.legend()
    if xlabel != '': plt.xlabel(xlabel)
    if ylabel != '': plt.ylabel(ylabel)
    if xlim[0] != 0 or xlim[1] != 0 : plt.xlim(xlim)
    if ylim[0] != 0 or ylim[1] != 0 : plt.ylim(ylim)
    if title != '': plt.suptitle(title)
    if grid: plt.grid()
    if save: plt.savefig(filename)
    if showPlot : plt.show()

def plotErrorData(values, labels, title = '', xlabel = '', ylabel = '', xlim = (0,0), ylim = (0,0), showPoints = True, grid = False, opacity = 1, save = False, filename = 'plot.pdf', showPlot = False):
    '''
        Draws a plot with error bars.

        Input parameters:
            - values [array( array( array(float), array(float), array((float,float)), string ), ... )] "dataset of points where each subarray points to a set of points. Subarray consists of 4 subarrays. First array is set of x coordinates, second is set of y coordinates, third is a set of tuples (float, float) where first parameters represents the error bar in the mius derection and the second parameter the error bar in the plus direction. The fourth parameter is for formating the point eg. 'ro' for red dots ",
            - labels [array(string)] "labels for each set of points. Supports LaTeX notation; to use it type between $ symbol eg. $\dot{V}_g = 0.887 m^3s^{-1}$",
            - title [string] "Title of the graph. Supports LaTeX notation; to use it type between $ symbol eg. $\dot{V}_g = 0.887 m^3s^{-1}$",
            - xlabel [string] "X axis label of the graph. Supports LaTeX notation; to use it type between $ symbol eg. $\dot{V}_g = 0.887 m^3s^{-1}$",
            - ylabel [string] "Y axis label of the graph. Supports LaTeX notation; to use it type between $ symbol eg. $\dot{V}_g = 0.887 m^3s^{-1}$",
            - xlim [tuple(int,int)] "X axis limit. Leave empty to include all data. eg. (0,100)",
            - ylim [tuple(int,int)] "Y axis limit. Leave empty to include all data. eg. (0,100)",
            - showPoints [bool] "True/False show points",
            - grid [bool] "True/False draw a grid",
            - opacity [float] "Opacity of lines",
            - save [bool] "True/False save plot as image",
            - filename [string] "name of he saved file. Extension can be set to .pdf, .jpg, .png ...",
            - showPlot [bool] "True/False show plot in external window"
    '''

    count = 0
    plt.clf()
    for i in values:
        name = labels[count]
        plt.errorbar(i[0],i[1],yerr = i[2], fmt = i[3],alpha = opacity,label=labels[count],capsize=3)
        if showPoints: plt.plot(i[0],i[1],'k^')
        count += 1
    plt.legend()
    if xlabel != '': plt.xlabel(xlabel)
    if ylabel != '': plt.ylabel(ylabel)
    if xlim[0] != 0 or xlim[1] != 0 : plt.xlim(xlim)
    if ylim[0] != 0 or ylim[1] != 0 : plt.ylim(ylim)
    if title != '': plt.suptitle(title)
    if grid: plt.grid()
    if save: plt.savefig(filename)
    if showPlot : plt.show()