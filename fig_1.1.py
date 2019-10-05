import numpy as np
import scipy as sp
import random
import math
import sympy
import pandas as pd
import rebind

b = 201.48  # 视角与入射角的方位夹角（逆时针），上下对称，只考虑pi方位
b = b / 180 * np.pi

G = 0.5
Ha = 0.3
Hb = 0
lamtai = 1  # 参数二 尼尔逊参数

# 各波段天空散射光比例
beltas = [0.454, 0.368, 0.282, 0.261, 0.257, 0.253, 0.248, 0.237, 0.217, 0.196]

# 植被反射光比例
Rcs = [0.0637, 0.135770, 0.084716, 0.152786, 0.202540, 0.262378, 0.401258, 0.675309, 0.812232, 0.825875]

# 土壤反射光比例
Rgs = [0.100541, 0.151270, 0.216372, 0.237683, 0.243111, 0.246032, 0.249314, 0.263932, 0.284605, 0.309330]

Rncs = [0.045, 0.09, 0.03, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

w1 = [0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.8, 0.8] # 参数三 单叶片反照率
belta = [0.454, 0.368, 0.282, 0.261, 0.257, 0.253, 0.248, 0.237, 0.217, 0.196]
qg = [0.100541, 0.151270, 0.216372, 0.237683, 0.243111, 0.246032, 0.249314, 0.263932, 0.284605, 0.309330]

Rm = [None] * 10 # 穿出和吸收

for LAIa in np.arange(5.1, 5.2, 0.1): # 参数一 实际LAI
    length_min = 0.00000001
    length_max = 1
    v_angle_min = 0
    v_angle_max = 0
    v_interval = 5
    s_angle_min = -60
    s_angle_max = 60
    s_interval = 5
    cons = -lamtai * G * LAIa
    
    p = 0.7 * math.exp(0.01 * lamtai * LAIa) - 0.66 * math.exp(-0.8 * lamtai * LAIa)
    for i in range(10):
        Gi = 0.5
        ceita = 60.21 # 参数四 太阳入射天顶角
        ui = math.cos(math.pi / 180 * ceita)
        i0 = 1 - math.exp(-lamtai * Gi * LAIa / ui) # 直射光拦截概率
        ia = 1 - math.exp(-math.pow((lamtai * LAIa), 0.9) * 0.8)
        m1 = 0.5 * (1 - belta[i]) * i0 * math.pow((w1[i]), 2) * p * (1 - p) / (1 - w1[i] * p)
        m1 = round(m1, 5)

        S1 = math.exp(-0.5867 * LAIa)
        m2 = belta[i] * (1 - S1) * ia / 2 * math.pow((w1[i]), 2) * p * (1 - p) / (1 - w1[i] * p)
        m2 = round(m2, 5)

        Sbs = i0 * math.pow(w1[i], 2) * m1 * (1 - m1) / (1 - w1[i] * m1)
        rc = (0.5 / ia) * Sbs
        m3 = (1 - belta[i]) * (1 - i0) * qg[i] * ia * rc / (1 - qg[i] * ia * rc) * (1 + qg[i] * (1 - ia))
        m3 = round(m3, 5)

        m4 = (1 - belta[i]) * i0 * rc * qg[i] / (1 - qg[i] * ia * rc) * (1 - ia + ia * rc)
        m4 = round(m4, 5)

        m5 = belta[i] * S1 * (1 - ia) * qg[i] * ia * rc / (1 - qg[i] * ia * rc) * (1 + qg[i] * (1 - ia))
        m5 = round(m5, 5)

        m6 = belta[i] * (1 - S1) * qg[i] * ia * rc / (1 - qg[i] * ia * rc) * (1 - ia * ia * rc)
        m6 = round(m6, 5)

        Tm = m1 + m2 + m3 + m4 + m5 + m6 # 总的多次散射项
        Rm[i] = round(Tm, 6)
        #print(round(Tm, 6))

#print(Rm)
    n=100000

    count1 = int((v_angle_max + v_interval - v_angle_min) / v_interval)
    lv = [None] * count1
    av = [None] * count1
    rvs = [None] * count1
    rvl = [None] * count1
    # 计算每个角度的聚集指数
    for m in np.arange(v_angle_min, v_angle_max + v_interval, v_interval):
        j = int((m - v_angle_min) / v_interval) # j 0-12
        lv[j] = -cons / math.cos(m / 180 * math.pi)
        av[j] = (0.5 * (Ha + Hb) * LAIa / Ha * math.tan(m / 180 * math.pi))
        rvs[j] = (math.sqrt(lv[j] * math.cos(m / 180 * math.pi)) / math.pi)
        rvl[j] = (rvs[j] / math.cos(m / 180 * math.pi))

    count2 = int((s_angle_max + s_interval - s_angle_min) / s_interval)
    ai = [None] * count2
    li = [None] * count2
    rs = [None] * count2
    rl = [None] * count2
    # 计算入射方向椭圆中心点和长短半轴
    for m in np.arange(s_angle_min, s_angle_max + s_interval, s_interval):
        j = int((m - s_angle_min) / s_interval)
        li[j] = -cons / math.cos(m / 180 * math.pi)
        ai[j] = 0.5 * (Ha + Hb) * LAIa / Ha * math.tan(m / 180 * math.pi)
        rs[j] = math.sqrt(li[j] * math.cos(m / 180 * math.pi)) / math.pi
        rl[j] = rs[j] / abs(math.cos(m / 180 * math.pi))
    
    length_lv = len(lv) * len(ai)
    K_ratio = np.zeros((length_lv, 8 + len(belta)))
    
    leaf_ratio0 = 0.99999999
    nonleaf_ratio0 = 0.00000001

    t = np.arange(0, 2 * math.pi, 0.01)
    lent = len(t)
    S = [None] * 13
    x0 = [None] * lent
    y0 = [None] * lent
    x1 = [None] * lent
    y1 = [None] * lent
    x2 = [None] * lent
    y2 = [None] * lent
    
    for m in np.arange(v_angle_min, v_angle_max + v_interval, v_interval):
        for tt in np.arange(s_angle_min, s_angle_max + s_interval, s_interval):
            j = int((m - v_angle_min) / v_interval)
            height = length_max * math.exp(cons) / length_max
            number5 = 0
            
            for i in range(n):
                x_s = (length_max - length_min) * random.random() + length_min
                y_s = height * random.random()
                if y_s < x_s * math.exp(cons / x_s):
                    number5 = number5 + 1
            
            S[j] = number5 / n
            kk = int((tt - s_angle_min) / s_interval)
            #t = np.arange(0, 2 * math.pi, 0.01)
            #lent = len(t)
            # mark
            x0 = rvl[j] * np.cos(math.pi / 180 * t) + av[j]
            y0 = rvs[j] * np.sin(math.pi / 180 * t)
            x1 = rl[kk] * np.cos(math.pi / 180 * t) + ai[kk]
            y1 = rs[kk] * np.sin(math.pi / 180 * t)
            x2 = x0 * np.cos(math.pi / 180 * b) - y0 * np.sin(math.pi / 180 * b)
            y2 = y0 * np.cos(math.pi / 180 * b) + x0 * np.sin(math.pi / 180 * b)

            x_min = [min(x1), min(x2)]
            x_min = min(x_min) - 0.5
            y_min = [min(y1), min(y2)]
            y_min = min(y_min) - 0.5
            x_max = [max(x1), max(x2)]
            x_max = max(x_max) + 0.5
            y_max = [max(y1), max(y2)]
            y_max = max(y_max) + 0.5 # 确定投点范围
            
            xx = sympy.Symbol('x')
            yy = sympy.Symbol('y')
            f1 = (xx - ai[kk]) ** 2 / (rl[kk] ** 2) + yy ** 2 / (rs[kk] ** 2) - 1
            f2 = (xx * sympy.cos(b) + yy * sympy.sin(b) -av[j]) ** 2 / (rvl[j] ** 2) + (yy * sympy.cos(b) - xx * sympy.sin(b)) ** 2 / (rvs[j] ** 2) -1
            result = sympy.solve([f1, f2], [xx, yy])
            X = []
            Y = []
            for i in result:
                if isinstance(i[0],complex) == False and isinstance(i[1],complex) == False:
                    X.append(i[0])
                    Y.append(i[1])
            
            if X:  # 判断X是否为空列表，也即没有实数解的不参与投点实验
                if X[0] != X[1]:
                    k = sympy.Symbol('k')
                    u = sympy.Symbol('u')
                    f3 = k * X[0] - Y[0] + u
                    f4 = k * X[1] - Y[1] + u
                    result1 = sympy.solve([f3, f4], [k, u])
                    K = result1.get(k)
                    U = result1.get(u)

                    # MC投点实验
                    t0 = 0
                    t1 = 0  # 交叠椭圆中属于入射椭圆的部分
                    t2 = 0  # 交叠椭圆中属于观测椭圆的部分

                    for i in np.arange(n):
                        x = (x_max - x_min) * random.random() + x_min
                        y = (y_max - y_min) * random.random() + y_min
                        if (x - ai[kk]) ** 2 / (rl[kk] ** 2) + y ** 2 /(rs[kk] ** 2) <= 1 and (x * math.cos(b) + y * math.sin(b) - av[j]) ** 2 / (rvl[j] ** 2) + (y * math.cos(b) - x * math.sin(b)) ** 2 / (rvs[j] ** 2) <= 1:
                            t0 = t0 + 1
                            if max(y1) < max(y2):
                                if K * x + U - y < 0:
                                    t1 = t1 + 1
                                else:
                                    t2 = t2 + 1
                            else:
                                if K * x + U - y < 0:
                                    t2 = t2 + 1
                                else:
                                    t1 = t1 + 1
                    
                    lci = t1 / n *(x_max - x_min) * (y_max - y_min) * math.cos(tt / 180 * math.pi)  # 存储lci
                    lcv = t2 / n *(x_max - x_min) * (y_max - y_min) * math.cos(m / 180 * math.pi)  # 存储lcv
                    lc = lci + lcv
                else:
                    t0 = 0
                    t1 = 0
                    t2 = 0
                    for i in np.arange(n):
                        x = (x_max - x_min) * random.random() + x_min
                        y = (y_max - y_min) * random.random() + y_min
                        if (x - ai[kk]) ** 2 / (rl[kk] ** 2) + y ** 2 /(rs[kk] ** 2) <= 1 and (x * math.cos(b) + y * math.sin(b) - av[j]) ** 2 / (rvl[j] ** 2) + (y * math.cos(b) - x * math.sin(b)) ** 2 / (rvs[j] ** 2) <= 1:
                            t0 = t0 + 1
                            if max(x1) > max(x2):
                                if x < X[0]:
                                    t1 = t1 + 1
                                else:
                                    t2 = t2 + 1
                            else:
                                if x < X[0]:
                                    t2 = t2 + 1
                                else:
                                    t1 = t1 + 1
                    lci = t1 / n * (x_max - x_min) * (y_max - y_min) * math.cos(tt / 180 * math.pi)  # 存储lci
                    lcv = t2 / n * (x_max - x_min) * (y_max - y_min) * math.cos(m / 180 * math.pi)  # 存储lcv
                    lc = lci + lcv
            else:
                lc = 0
        
            Kc = 1 - math.exp(-lc)
            Kcl = Kc * leaf_ratio0
            Kcnl = Kc * nonleaf_ratio0
            Kcnl_1 = Kc - Kcl -Kcnl
            Kg = math.exp(lc - li[kk] - lv[j])
            Kz = math.exp(-lv[j]) - Kg
            Kt = (1 - Kc - Kg - Kz) * leaf_ratio0
            Kcnl_1 = Kcnl_1 + (1 - Kc - Kg - Kz) * (1-leaf_ratio0)

            K_ratio[len(ai) * j + kk, 0] = m  # 存储观测角
            K_ratio[len(ai) * j + kk, 1] = tt  # 存储太阳天顶角
            K_ratio[len(ai) * j + kk, 2] = Kcl
            K_ratio[len(ai) * j + kk, 3] = Kcnl
            K_ratio[len(ai) * j + kk, 4] = LAIa# Kcnl_1
            K_ratio[len(ai) * j + kk, 5] = Kg
            K_ratio[len(ai) * j + kk, 6] = Kz
            K_ratio[len(ai) * j + kk, 7] = Kt

            K_ratio[len(ai) * j + kk, 8] = (1 - belta[0]) * (Kcl * Rcs[0] + Kg * Rgs[0]) + (1 - S[j]) * Kt * Rcs[0] + S[j] * belta[0] * Kz * Rgs[0] + (1 - belta[0]) * Kcnl * Rncs[0] + (1 - S[j]) * belta[0] * Kcnl_1 * Rncs[0] + Rm[0]
            K_ratio[len(ai) * j + kk, 9] = (1 - belta[1]) * (Kcl * Rcs[1] + Kg * Rgs[1]) + (1 - S[j]) * Kt * Rcs[1] + S[j] * belta[1] * Kz * Rgs[1] + (1 - belta[1]) * Kcnl * Rncs[1] + (1 - S[j]) * belta[1] * Kcnl_1 * Rncs[1] + Rm[1]
            K_ratio[len(ai) * j + kk, 10] = (1 - belta[2]) * (Kcl * Rcs[2] + Kg * Rgs[2]) + (1 - S[j]) * Kt * Rcs[2] + S[j] * belta[2] * Kz * Rgs[2] + (1 - belta[2]) * Kcnl * Rncs[2] + (1 - S[j]) * belta[2] * Kcnl_1 * Rncs[2] + Rm[2]
            K_ratio[len(ai) * j + kk, 11] = (1 - belta[3]) * (Kcl * Rcs[3] + Kg * Rgs[3]) + (1 - S[j]) * Kt * Rcs[3] + S[j] * belta[3] * Kz * Rgs[3] + (1 - belta[3]) * Kcnl * Rncs[3] + (1 - S[j]) * belta[3] * Kcnl_1 * Rncs[3] + Rm[3]
            K_ratio[len(ai) * j + kk, 12] = (1 - belta[4]) * (Kcl * Rcs[4] + Kg * Rgs[4]) + (1 - S[j]) * Kt * Rcs[4] + S[j] * belta[4] * Kz * Rgs[4] + (1 - belta[4]) * Kcnl * Rncs[4] + (1 - S[j]) * belta[4] * Kcnl_1 * Rncs[4] + Rm[4]
            K_ratio[len(ai) * j + kk, 13] = (1 - belta[5]) * (Kcl * Rcs[5] + Kg * Rgs[5]) + (1 - S[j]) * Kt * Rcs[5] + S[j] * belta[5] * Kz * Rgs[5] + (1 - belta[5]) * Kcnl * Rncs[5] + (1 - S[j]) * belta[5] * Kcnl_1 * Rncs[5] + Rm[5]
            K_ratio[len(ai) * j + kk, 14] = (1 - belta[6]) * (Kcl * Rcs[6] + Kg * Rgs[6]) + (1 - S[j]) * Kt * Rcs[6] + S[j] * belta[6] * Kz * Rgs[6] + (1 - belta[6]) * Kcnl * Rncs[6] + (1 - S[j]) * belta[6] * Kcnl_1 * Rncs[6] + Rm[6]
            K_ratio[len(ai) * j + kk, 15] = (1 - belta[7]) * (Kcl * Rcs[7] + Kg * Rgs[7]) + (1 - S[j]) * Kt * Rcs[7] + S[j] * belta[7] * Kz * Rgs[7] + (1 - belta[7]) * Kcnl * Rncs[7] + (1 - S[j]) * belta[7] * Kcnl_1 * Rncs[7] + Rm[7]
            K_ratio[len(ai) * j + kk, 16] = (1 - belta[8]) * (Kcl * Rcs[8] + Kg * Rgs[8]) + (1 - S[j]) * Kt * Rcs[8] + S[j] * belta[8] * Kz * Rgs[8] + (1 - belta[8]) * Kcnl * Rncs[8] + (1 - S[j]) * belta[8] * Kcnl_1 * Rncs[8] + Rm[8]
            K_ratio[len(ai) * j + kk, 17] = (1 - belta[9]) * (Kcl * Rcs[9] + Kg * Rgs[9]) + (1 - S[j]) * Kt * Rcs[9] + S[j] * belta[9] * Kz * Rgs[9] + (1 - belta[9]) * Kcnl * Rncs[9] + (1 - S[j]) * belta[9] * Kcnl_1 * Rncs[9] + Rm[9]

    writer = pd.ExcelWriter(r'./datas/' +str(round(LAIa, 2)) + '.xlsx')
    data_df = pd.DataFrame(K_ratio)
    data_df.to_excel(writer, 'page_1', float_format='%.5f')
    writer.save()
            
                    
rebind.rebinds(r'./datas')            

