# coding:utf-8
import numpy as np
from numpy.lib.function_base import append
import pandas as pd
import math

class FreeField(object):

    def __init__(self, x):
        self.f1 = x[0]
        self.f2 = x[1]


    # 時刻歴応答解析
    def resp(self):
        f1 = self.f1
        f2 = self.f2

        model = pd.read_csv("model.csv")
        h1 = model["h"][0]
        h2 = model["h"][0]

        size = len(model["M"])
        # 質量マトリクス
        m = np.diag(list(map(lambda x: x / 9.80665, model["M"])))
        # 剛性マトリクス
        k = np.zeros((size, size))
        kdata = size -1
        for i in range(kdata):
            i1 = i
            i2 = i + 1
            k[i1][i1] = k[i1][i1] + model["K"][i]
            k[i2][i2] = k[i2][i2] + model["K"][i]
            k[i1][i2] = k[i1][i2] - model["K"][i]
            k[i2][i1] = k[i2][i1] - model["K"][i]
        # 減衰マトリクス(レイリー減衰)
        w1 = 2.0 * np.pi * f1
        w2 = 2.0 * np.pi * f2
        alpha = (2.0 * w1 * w2 * (h2 * w1 - h1 * w2)) / (w1 ** 2 - w2 ** 2)
        beta = (2.0 * (h1 * w1 - h2 * w2)) / (w1 ** 2 - w2 ** 2)
        c = np.array(alpha * m + beta * k)

        # 底面ダンパー(2E入力用)
        c[kdata][kdata] = c[kdata][kdata] + model["C"][0]

        # 入力波読み込み
        df = pd.read_csv(model["accfile"][0])
        acc0 = df["acc"]
        dt = 0.01
        beta = 0.25
        unit_vector = np.ones(size)
        pre_acc0 = 0.0
        time = 0.0
        dis = np.zeros(size)
        vel = np.zeros(size)
        acc = np.zeros(size)
        ddis = np.zeros(size)
        dvel = np.zeros(size)
        dacc = np.zeros(size)
        acc_history = {}
        for i in range(0, size):
            acc_history[i] = []
        time_history = []
        # Newmarkβ法による数値解析（増分変位による表現）
        for i in range(0, len(acc0)):
            kbar = k + (1.0/(2.0*beta*dt)) * c + (1.0/(beta*dt**2.0)) * m
            dp1 = -1.0 * m.dot(unit_vector) * (acc0[i] - pre_acc0)
            dp2 = m.dot((1.0/(beta*dt))*vel + (1.0/(2.0*beta))*acc)
            dp3 = c.dot((1.0/(2.0*beta))*vel + (1.0/(4.0*beta)-1.0)*acc*dt)
            dp = dp1 + dp2 + dp3
            ddis = np.linalg.inv(kbar).dot(dp)
            dvel = (1.0/(2.0*beta*dt))*ddis - (1.0/(2.0*beta))*vel - ((1.0/(4.0*beta)-1.0))*acc*dt
            dacc = (1.0/(beta*dt**2.0))*ddis - (1.0/(beta*dt))*vel - (1.0/(2.0*beta))*acc
            dis += ddis
            vel += dvel
            acc += dacc
            acc_abs = acc + [acc0[i] for n in range(0,size)]
            [acc_history[i].append(x) for i, x in enumerate(acc_abs)]
            time_history.append(time)
            time += dt
            pre_acc0 = acc0[i]

        # 評価対象として最上部の応答スペクトルを計算する
        spec = self.duhm_spec(acc_history[0])
        return(spec)


    def duhm_spec(self, acc):
        spec = np.empty(0)
        dlta = (math.log(5.0) - math.log(0.02)) / float(300 - 1)
        for i in range(1, 300 + 1):
            tmp = math.log(0.02) + dlta * float(i - 1)
            period = math.exp(tmp)
            zmax = self.duhm(0.01, 0.05, period, acc)
            spec = np.append(spec, zmax)
        return(spec)


    def duhm(self, dt, h, T, a):
        """duhamel integral"""
        w = 2.0 * math.pi / T
        w2 = w * w
        hw = h * w
        wd = w * math.sqrt( 1.0 - h * h )
        wdt = wd * dt
        e = math.exp( -hw * dt )
        cwdt = math.cos( wdt )
        swdt = math.sin( wdt )
        e11 = e * ( cwdt - hw * swdt / wd )
        e12 = -e * ( wd * wd + hw * hw ) * swdt / wd
        e21 = e * swdt / wd
        e22 = e * ( cwdt + hw * swdt / wd )
        ss = - hw * swdt - wd * cwdt
        cc = - hw * cwdt + wd * swdt
        s1 = ( e * ss + wd ) / w2
        c1 = ( e * cc + hw ) / w2
        s2 = ( e * dt * ss + hw * s1 + wd * c1 ) / w2
        c2 = ( e * dt * cc + hw * c1 - wd * s1 ) /w2
        s3 = dt * s1 - s2
        c3 = dt * c1 - c2
        g11 = ( -hw * s3 + wd * c3 ) / wdt
        g12 = ( -hw * s2 + wd * c2 ) / wdt
        g21 = s3 / wdt
        g22 = s2 / wdt
        dx = 0.0
        x = 0.0
        accmax = 0.0
        for m in range(1, len(a)):
            dxf = dx
            xf = x
            ddym = a[ m ]
            ddyf = a[ m - 1 ]
            dx = e11 * dxf + e12 * xf + g11 * ddym + g12 * ddyf
            x = e21 * dxf + e22 * xf + g21 * ddym + g22 * ddyf
            ddx = 2.0 * hw * dx + w2 * x
            if abs(ddx) > abs(accmax):
                accmax = abs(ddx)

        return(accmax)
