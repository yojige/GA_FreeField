# coding:utf-8

import numpy as np
from numpy import random as rand
import pandas as pd
from FreeField import FreeField
import matplotlib.pyplot as plt

from PIL import Image
import glob

class GeneticAlgorithm(object):
    """遺伝的アルゴリズム"""

    def __init__(self):
        """初期化"""

        """定数"""
        # 染色体数
        self.N = 2
        # 個体数
        self.M = 10
        # エリート選択個数
        self.n_elite = 3
        self.p_mutation = 0.10

        """変数"""
        # 第g世代評価降順インデックス
        self.val_sorted_index = []
        # 選択された遺伝子のインデックス
        self.selected_index = []
        # 次世代遺伝子
        self.next_generation = []
        # 各世代の最良応答スペクトル
        self.best_spec = []

        """初期値"""
        # 第g世代遺伝子群
        self.X_g = np.array([[rand.uniform(0.2,50.0),       # f1
                             rand.uniform(0.2,50.0)]        # f2
                             for i in range(self.M)])

        """ターゲットスペクトル"""
        self.df_target_spec = pd.read_csv("shake.csv")


    def assess(self):
        """評価"""
        # ターゲットスペクトルとの差分2乗和を評価値とする
        FreeField_spec = np.array([FreeField(x).resp() for x in self.X_g])
        R = []
        for spec in FreeField_spec:
            R.append(np.sum((self.df_target_spec["shake"] - spec) ** 2))
        R = np.array(R)
        self.val_sorted_index = np.argsort(R)[::1]
        self.best_spec.append(FreeField_spec[self.val_sorted_index[0]])


    def select(self):
        """選択"""
        # エリート選択で N 個
        elite_index = self.val_sorted_index[0:self.n_elite]
        self.selected_index = np.r_[elite_index]


    def crossover1p(self):
        """交叉"""
        ng = []
        for i in range(self.M - self.n_elite):
            p_index = rand.choice(self.selected_index, 2, replace=False)
            new_g = np.r_[self.X_g[p_index[0]][0], self.X_g[p_index[1]][1]]
            ng.append(new_g)

        ng = np.array(ng)
        self.next_generation = np.r_[ng, self.X_g[self.selected_index[0:self.n_elite]]]

    def mutation(self):
        """突然変異"""
        # 確率的に選ばれた染色体をランダム値に変更
        # 突然変異を起こす染色体の数
        n_mutation = rand.binomial(n=self.M * self.N, p=self.p_mutation)

        # 突然変異
        for i in range(n_mutation):
            m = rand.randint(self.M)
            n = rand.randint(self.N)
            self.next_generation[m][n] = rand.uniform(0.2,50.0)

    def alternation(self):
        """世代交代"""
        self.X_g = self.next_generation


if __name__ == '__main__':

    """設定"""
    # 計算世代
    G = 50

    """計算"""
    ga = GeneticAlgorithm()
    for g in range(G):
        # 評価
        ga.assess()
        # 選択
        ga.select()
        # 交叉
        ga.crossover1p()
        # 突然変異
        ga.mutation()
        # 世代交代
        ga.alternation()

    for i, spec in enumerate(ga.best_spec):
        # グラフ描画
        plt.plot(ga.df_target_spec["t"], ga.df_target_spec["shake"], label="Target")
        plt.plot(ga.df_target_spec["t"], spec, label="GA")
        plt.title("GA = "+str(i))
        plt.xlabel("Period(sec)")
        plt.ylabel("Acceleration Response Spectrum $(m/s^{2})$")
        plt.xscale("log")
        plt.ylim(0.0, 20.0)
        plt.legend()
        plt.grid()
        png = "spec_{:0>2d}.png".format(i)
        plt.savefig(png)
        plt.clf()

    # GIF 作成
    files = sorted(glob.glob('*.png'))  
    images = list(map(lambda file : Image.open(file) , files))
    images[0].save('GA.gif' , save_all = True , append_images = images[1:] , duration = 500)
