"""
    Batuhan Çalışkan    2309805
    Alişan Yıldırım    2172161
"""
import math
import random

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os


def MedianFilter(Input_Image, size, color):
    Image_Height = Input_Image.shape[0]
    Image_Width = Input_Image.shape[1]

    New_Image = np.zeros((Image_Height, Image_Width), np.int32)

    Arr = np.zeros([size* size], np.int32)
    for y in range(0, Image_Height):
        for x in range(0, Image_Width):
            for i in range(0, size):
                for j in range(0, size):
                    if not ((x + (i - size//2)) < 0 or (x + (i - size//2)) > (Image_Width - 1) or (y + (j - size//2)) < 0 or (y + (j - size//2)) > (Image_Height -1)):
                        Arr[j*size + i] = (Input_Image[y + (j - size//2)][x + (i - size//2)][color])
            Arr.sort()
            New_Image[y][x] = np.median(Arr)

    return New_Image

def EdgeClamper(Input_Image, val):
    Image_Height = Input_Image.shape[0]
    Image_Width = Input_Image.shape[1]

    for x in range(0, Image_Width):
        for y in range(0, Image_Height):
            if Input_Image[y][x] < val:
                Input_Image[y][x] = 0
            else:
                Input_Image[y][x] = 255
    return Input_Image

def Convolution(Input_Image, filter, color):
    Image_Height = Input_Image.shape[0]
    Image_Width = Input_Image.shape[1]
    filter_size = len(filter)
    half_filter_size = ((filter_size-1) // 2)
    New_Image = np.zeros((Image_Height, Image_Width))
    for x in range(0, Image_Width):
        for y in range(0, Image_Height):
            val = 0
            for i in range(0, filter_size):
                for j in range(0, filter_size):
                    if (x + (i - half_filter_size)) < 0 or (x + (i - half_filter_size)) > (Image_Width - 1) or (y + (j - half_filter_size)) < 0 or (y + (j - half_filter_size)) > (Image_Height -1):
                        val += 0
                    else:
                        val += filter[j][i] * (Input_Image[y + (j - half_filter_size)][x + (i - half_filter_size)][color])
            New_Image[y][x] = val
    return New_Image

def FindIndex(Arr, val):
    for x in range(0, len(Arr)):
        if Arr[x] == val:
            return x
    return val

def MakeDir(output_path):
    path_list=output_path.split("/")
    writen_path=""
    for path in path_list:
        if path=="":
            continue
        if path=="." or path=="~":
            writen_path+=path+"/"
            continue
        writen_path+=path
        if not os.path.exists(writen_path):
            os.mkdir(writen_path)
        writen_path+="/"

def part1 ( input_img_path , output_path , m , s ):
    if output_path[-1]!="/":
        output_path+="/"
    if not os.path.exists(output_path):
        MakeDir(output_path)

    Input_Image = cv.imread(input_img_path)
    Image_Height = Input_Image.shape[0]
    Image_Width = Input_Image.shape[1]

    Histogram = np.zeros([256], np.int32)
    for x in range(0, Image_Height):
        for y in range(0, Image_Width):
            Histogram[Input_Image[x][y][0]] += 1

    plt.bar( np.arange(len(Histogram)), Histogram )
    plt.savefig(output_path + "original_histogram.png")

    Histogram_Cumilative = np.zeros([256], np.int32)
    Histogram_Cumilative[0] = Histogram[0]
    for x in range(1, 256):
            Histogram_Cumilative[x] += Histogram_Cumilative[x-1] + Histogram[x]

    Transform_1 = np.zeros([256], np.int32)
    for x in range(0, 256):
        Transform_1[x] = round( (255 / (Image_Height * Image_Width) ) * Histogram_Cumilative[x])

     # Mixture of Two Gaussians
    Histogram_G = np.zeros([256], np.int32)
    for x in range(0, len(m)):
        Histogram_G = np.add(Histogram_G, np.random.normal(m[x], s[x], 256))

    plt.bar(np.arange(len(Histogram_G)), Histogram_G)
    plt.savefig(output_path + "gaussian_histogram.png")

    Histogram_G_Cumilative = np.zeros([256], np.int32)
    Histogram_G_Cumilative[0] = Histogram_G[0]
    for x in range(1, 256):
        Histogram_G_Cumilative[x] += Histogram_G_Cumilative[x - 1] + Histogram_G[x]

    Transform_2 = np.zeros([256], np.int32)
    for x in range(0, 256):
        Transform_2[x] = round((255 / np.max(Histogram_G_Cumilative)) * Histogram_G_Cumilative[x])

    for x in range(0, Image_Height):
         for y in range(0, Image_Width):
             Input_Image[x][y] = FindIndex(Transform_2, Transform_1[Input_Image[x][y][0]])

    cv.imwrite(output_path + "matched_image.png", Input_Image)

    Histogram = np.zeros([256], np.int32)
    for x in range(0, Image_Height):
        for y in range(0, Image_Width):
            Histogram[Input_Image[x][y][0]] += 1

    plt.bar(np.arange(len(Histogram)), Histogram)
    plt.savefig(output_path + "matched_image_histogram.png")

def the1_convolution (input_img_path , filter):
    Input_Image = cv.imread(input_img_path)
    return Convolution(Input_Image, filter, 0)

def part2 ( input_img_path , output_path ) :
    if output_path[-1]!="/":
            output_path+="/"
    if not os.path.exists(output_path):
        MakeDir(output_path)

    Input_Image = cv.imread(input_img_path)
    Image_Height = Input_Image.shape[0]
    Image_Width = Input_Image.shape[1]

    Smoothed_Image = cv.blur(Input_Image, (15, 15))

    Edge_Image_x_1 = Convolution(Smoothed_Image, [[-1, 0, 1],
                                                [-2, 0, 2],
                                                [-1, 0, 1]], 0)

    Edge_Image_y_1 = Convolution(Smoothed_Image, [[ 1,  2,  1],
                                                [ 0,  0,  0],
                                                [-1, -2, -1]], 0)

    Edge_Image_x_2 = Convolution(Smoothed_Image, [[1, 0, -1],
                                                [2, 0, -2],
                                                [1, 0, -1]], 0)

    Edge_Image_y_2 = Convolution(Smoothed_Image, [[ -1,  -2,  -1],
                                                [ 0,  0,  0],
                                                [1, 2, 1]], 0)

    Edge_Image_x_1 = EdgeClamper(Edge_Image_x_1, 20)
    Edge_Image_y_1 = EdgeClamper(Edge_Image_y_1, 20)
    Edge_Image_x_2 = EdgeClamper(Edge_Image_x_2, 20)
    Edge_Image_y_2 = EdgeClamper(Edge_Image_y_2, 20)

    Output_Image = np.zeros((Image_Height, Image_Width))

    for x in range(0, Image_Width):
        for y in range(0, Image_Height):
            Output_Image[y][x] = Edge_Image_x_1[y][x] + Edge_Image_y_1[y][x] + Edge_Image_x_2[y][x] + Edge_Image_y_2[y][x]

    Output_Image = EdgeClamper(Output_Image, 1)

    cv.imwrite(output_path + "edges.png", Output_Image)
    return

def enhance_3 ( path_to_3 , output_path ) :
    if output_path[-1]!="/":
        output_path+="/"
    if not os.path.exists(output_path):
        MakeDir(output_path)

    Input_Image = cv.imread(path_to_3)
    Image_Height = Input_Image.shape[0]
    Image_Width = Input_Image.shape[1]
    Output_Image_R = Convolution(Input_Image, [  [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
                                                 [1 / 25, 1 / 25, 2 / 25, 1 / 25, 1 / 25],
                                                 [1 / 25, 2 / 25, 3 / 25, 2 / 25, 1 / 25],
                                                 [1 / 25, 1 / 25, 2 / 25, 1 / 25, 1 / 25],
                                                 [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25]], 0)
    Output_Image_G = Convolution(Input_Image, [  [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
                                                 [1 / 25, 1 / 25, 2 / 25, 1 / 25, 1 / 25],
                                                 [1 / 25, 2 / 25, 3 / 25, 2 / 25, 1 / 25],
                                                 [1 / 25, 1 / 25, 2 / 25, 1 / 25, 1 / 25],
                                                 [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25]], 1)
    Output_Image_B = Convolution(Input_Image, [  [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
                                                 [1 / 25, 1 / 25, 2 / 25, 1 / 25, 1 / 25],
                                                 [1 / 25, 2 / 25, 3 / 25, 2 / 25, 1 / 25],
                                                 [1 / 25, 1 / 25, 2 / 25, 1 / 25, 1 / 25],
                                                 [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25]], 2)

    Output_Image = np.zeros((Image_Height, Image_Width, 3))
    for x in range(0, Image_Width):
        for y in range(0, Image_Height):
            Output_Image[y][x][0] = Output_Image_R[y][x]
            Output_Image[y][x][1] = Output_Image_G[y][x]
            Output_Image[y][x][2] = Output_Image_B[y][x]

    cv.imwrite(output_path + "enhanced.png", Output_Image)
    return Output_Image

def enhance_4 ( path_to_4 , output_path ) :
    if output_path[-1]!="/":
        output_path+="/"
    if not os.path.exists(output_path):
        MakeDir(output_path)

    Input_Image = cv.imread(path_to_4)
    Image_Height = Input_Image.shape[0]
    Image_Width = Input_Image.shape[1]
    Output_Image_R = Convolution(Input_Image, [[1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
                                               [1 / 25, 1 / 25, 2 / 25, 1 / 25, 1 / 25],
                                               [1 / 25, 2 / 25, 3 / 25, 2 / 25, 1 / 25],
                                               [1 / 25, 1 / 25, 2 / 25, 1 / 25, 1 / 25],
                                               [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25]], 0)
    Output_Image_G = Convolution(Input_Image, [[1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
                                               [1 / 25, 1 / 25, 2 / 25, 1 / 25, 1 / 25],
                                               [1 / 25, 2 / 25, 3 / 25, 2 / 25, 1 / 25],
                                               [1 / 25, 1 / 25, 2 / 25, 1 / 25, 1 / 25],
                                               [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25]], 1)
    Output_Image_B = Convolution(Input_Image, [[1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25],
                                               [1 / 25, 1 / 25, 2 / 25, 1 / 25, 1 / 25],
                                               [1 / 25, 2 / 25, 3 / 25, 2 / 25, 1 / 25],
                                               [1 / 25, 1 / 25, 2 / 25, 1 / 25, 1 / 25],
                                               [1 / 25, 1 / 25, 1 / 25, 1 / 25, 1 / 25]], 2)

    Output_Image = np.zeros((Image_Height, Image_Width, 3))
    for x in range(0, Image_Width):
        for y in range(0, Image_Height):
            Output_Image[y][x][0] = Output_Image_R[y][x]
            Output_Image[y][x][1] = Output_Image_G[y][x]
            Output_Image[y][x][2] = Output_Image_B[y][x]

    cv.imwrite(output_path + "enhanced1.png", Output_Image)

    Output_Image_R_M = MedianFilter(Input_Image, 3, 0)
    Output_Image_G_M = MedianFilter(Input_Image, 3, 1)
    Output_Image_B_M = MedianFilter(Input_Image, 3, 2)

    Output_Image_M = np.zeros((Image_Height, Image_Width, 3))
    for x in range(0, Image_Width):
        for y in range(0, Image_Height):
            Output_Image_M[y][x][0] = Output_Image_R_M[y][x]
            Output_Image_M[y][x][1] = Output_Image_G_M[y][x]
            Output_Image_M[y][x][2] = Output_Image_B_M[y][x]

    cv.imwrite(output_path + "enhanced2.png", Output_Image_M)

    return Output_Image_M

