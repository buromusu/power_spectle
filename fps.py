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
import cal_ps as ps
import image_sort as iso


def my_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return 0


if __name__ == '__main__':

    in_root = "/Volumes/buromusu2/CelebA-HQ1000" #元画像フォルダ
    csv_root = "/Volumes/buromusu2/cs" #csvがあるフォルダ
    out_root = "/Volumes/buromusu2/CelebA-HQ1000sorted" #上位画像を保存するフォルダ
    file_count = 10 #何件取り出すか
    my_mkdir(csv_root)
    my_mkdir(out_root)



    # init scores
    b_scores, a_scores = [], []

    # スコア計算
    filenames = sorted(os.listdir(in_root))
    for filename in tqdm(filenames, desc='calc score'):
        filepath = os.path.join(in_root, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        mag = ps.power_spectrum(img)
        mag_polar = ps.cartesian2polar(mag)
        a_score = ps.asymmetry_score(mag_polar)
        b_score = ps.broadness_score(mag_polar)
        a_scores.append(a_score); b_scores.append(b_score)

    #スコアを正規化
    a_scores = np.array(a_scores); b_scores = np.array(b_scores)
    b_scores_origine = np.array(b_scores)
    b_scores = np.log(b_scores)
    a_score_mean = a_scores.mean(); b_score_mean = b_scores.mean(); b_score_origine_mean = b_scores_origine.mean()
    a_score_std = a_scores.std(); b_score_std = b_scores.std(); b_score_origine_std = b_scores_origine.std()
    a_scores = (a_scores - a_score_mean) / a_score_std
    b_scores = (b_scores - b_score_mean) / b_score_std
    b_scores_origine = (b_scores_origine - b_score_origine_mean) / b_score_origine_std
    a_scores = a_scores + 2.0
    b_scores_origine = (-1) * b_scores_origine
    b_scores = np.log(b_scores_origine + 2.0)


    #total_score算出
    total_scores = ps.t_score(a_scores, b_scores, 18)

    #csvファイル作成
    score_hash = {}
    final_score = []
    for j in range(len(total_scores)):
        score_hash[total_scores[j]] = j
    for key in sorted(list(score_hash.keys()), reverse=True):
        final_score.append(score_hash[key])
    resultdf = pd.DataFrame(columns = ["file", "total_scores", "rank"])
    final = []
    for i in range(len(a_scores)):
        final.clear()
        final.append(filenames[i])
        final.append(total_scores[i])
        final.append(final_score[i])
        df1 = pd.DataFrame(final).T
        df1.index = [i]
        df1.columns = ["file", "total_scores", "rank"]
        resultdf = pd.concat([resultdf, df1])
    resultdf = resultdf.sort_values(by='rank')
    resultdf.to_csv(os.path.join(csv_root, "rank.csv"))


    #作成したcsvを用いてソート
    iso.file_sort(file_count, in_root, out_root, csv_root)
