# パワースペクトル指標の分析
import argparse, math, os
import numpy as np
import cv2
from tkinter import *
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from timeit import default_timer as timer
import shutil
from tqdm import tqdm
import csv
import pandas as pd
from scipy.stats import zscore
from scipy.stats import entropy

#スコア計算モジュール

#a_score計算
def asymmetry_score(image):
    """Measure approximate radial asymmetry."""
    score = mad(image / image.sum(axis=1)[:, None]).mean()
    return score

#b_score計算
def broadness_score(image):
    sum_r = np.sum(image, axis=1)
    score = sum_r.std()
    return score

#パワースペクトル作成
def power_spectrum(img):
    # 高速フーリエ変換(2次元)
    fimg = np.fft.fft2(img)
    # 第1象限と第3象限, 第2象限と第4象限を入れ替え
    fimg = np.fft.fftshift(fimg)
    # パワースペクトルの計算
    mag = 20 * np.log(np.abs(fimg))
    return mag

#極座標にする
def cartesian2polar(image):
    r_max = theta_max = min(image.shape) // 2 # 動径の定義域の最大値
    theta_coordinate, r_coordinate = np.meshgrid(np.arange(theta_max), np.arange(r_max))
    theta_rad = 2.0 * np.pi * (theta_coordinate / theta_max)
    origin = np.array(image.shape) / 2
    x = origin[0] + r_coordinate * np.cos(theta_rad)
    y = origin[1] + r_coordinate * np.sin(theta_rad)
    x, y = x.astype(np.uint16), y.astype(np.uint16)
    image_polar = image[x, y]
    return image_polar

#MAD計算
def mad(list):
    med = np.median(list, axis=1)
    newlist = []
    for i in range(len(list)):
        newlist.append(np.abs(list[i] - med[i]))
    mad = np.array(newlist).mean(axis=1)
    madn = mad / 0.675
    return madn


#total_score計算
def t_score(a_scores, b_scores, c):
    total_scores = 1000 + b_scores - c*0.01*a_scores
    return total_scores
