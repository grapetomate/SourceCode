# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 22:01:25 2016

@author: OmuraKazuki
"""

"""
プログラム説明
・ハードマージンL1SVMです(L2SVMというのもありますが勉強会ではやってません)
・オーダーはO(n^3)らしいです.データ数が増えると計算時間がとんでもなく長いです.
・サポートベクトルは赤色でプロットします
・仕様上,識別面の傾きの絶対値は約1以下です.つまり斜めか水平な分割しかしてくれません
・仕様上,識別面の切片も絶対値約1以下になってしまうので最後に補正をかけてます
・各クラスのデータ数を同じにしないとエラーが起こります
・完全な識別面が引けない分布のときにもエラーが起こります
・勉強会では,pの選び方はKKT条件を破るものから選ぶと言いましたが,実際にはKKT条件を破り,
 かつ|f(X[p])|が最大になるものを選ばないとうまく行きませんでした
"""

import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt

N = 200 #各クラスのデータ数
C = 3 #ハードマージンSVMでは使わない

#カーネル
def kernel(x, y):
    return x @ y.T

#識別関数f(x)
def f(X, T, mu, theta, x):
    w = X.T * mu * T #重み
    w = np.array([np.sum(w[0]), np.sum(w[1])])
    return  w @ x.T + theta

def theta_f(X, T, mu):
    S = np.array([]) #muの中から要素が0ではないものの番号を取り出したリスト
    for i in range(len(mu)):
        if mu[i] != 0:
            S = np.append(S, i) #Sに要素を追加
    S = np.array([np.int(i) for i in S])
    w = X.T * mu * T #重み
    w = np.array([np.sum(w[0]), np.sum(w[1])])
    theta = np.sum(T[i] - w.T @ X[i] for i in S) / len(S)
    return theta    
    
"""
SMO法
#1.KKT条件を満たさず,かつ|f(X[p])|が最大になる番号をpとする
#2.pに対して、F = |f(X[p])-f(X[q])| を最大とし、かつKKT条件を満たさないqを1つ選択
#3.p,qに対するmuの値(mu[p],mu[q])を更新
#4.更新した値およびp,qを返す
"""
def SMO(X, T, theta, mu):
    #1
    w = X.T * mu * T
    w = np.array([np.sum(w[0]), np.sum(w[1])])
    
    p = np.array([])
    for i in range(len(mu)):
        if mu[i] >= 0:
            if T[i] * (w.T @ X[i] + theta) >= 1:
                if mu[i] * (T[i] * (w.T @ X[i] + theta) - 1) == 0:
                    continue #KKT条件を全て満たすiは処理せずに次へ
        #KKT条件を満たさない値はリストに入れる
        p = np.append(p, i)   
    p = np.array([np.int(i) for i in p])
    
    #ｐが空なら反復が不要なので処理を打ち切り
    if len(p) == 0:
        return True
    
    #|f(X[p])|が最大となるものを選ぶ
    Fp = np.array([f(X, T, mu, theta, X[i]) for i in p]) #f(X[p])のリスト
    dp = {x : y for x, y in zip(Fp, p)}  #Fpの値とpの値を対応させた辞書作成       
    xp = np.nonzero(np.fabs(Fp) == max(np.fabs(Fp))) #|f(X[p])|の最大値を探索
    xp = np.int(xp[0][0]) #見つけた値を取り出す
    p = dp[Fp[xp]] #辞書から最大値に対応するリスト番号を取得
    
    #2
    #条件を満たすqを全て探索
    q = np.array([])
    for i in range(len(mu)):
        if i != p: #qはpと異なる値から選ぶ
            if mu[i] >= 0:
                if T[i] * (w.T @ X[i] + theta) >= 1:
                    if mu[i] * (T[i] * (w.T @ X[i] + theta) - 1) == 0:
                        continue #KKT条件を全て満たすiは処理せずに次へ
            #KKT条件を満たさない値はリストに入れる
            q = np.append(q, i) 
    q = np.array([np.int(i) for i in q])
    
    #qが空なら反復が不要なので処理を打ち切り
    if len(q) == 0:
        return True
    
    #全てのqに対してf(X[q])-f(X[p])を計算
    #f(X[q])-f(X[p])のリスト    
    Fq = np.array([f(X, T, mu, theta, X[i]) for i in q]) - f(X, T, mu, theta, X[p])    
    dq = {x : y for x, y in zip(Fq, q)} #Fqの値とqの値を対応させた辞書作成
    xq = np.nonzero(np.fabs(Fq) == max(np.fabs(Fq))) #f(X[q])-f(X[p])の最大値を探索    
    xq = np.int(xq[0][0]) #見つけた値を取り出す
    q = dq[Fq[xq]] #辞書から最大値に対応するリスト番号を取得   
    
    #3
    c = T[p] * mu[p] + T[q] * mu[q] #パラメータc
    #更新量dp
    dp =  (1 - T[p] * T[q] + T[p] * (f(X, T, mu, theta, X[q]) - f(X, T, mu, theta, X[p])))  \
            / (kernel(X[p], X[p]) - 2 * kernel(X[p], X[q]) + kernel(X[q], X[q]))
    #更新するときの条件分岐        
    if T[p] == T[q]:
        if mu[p] + dp > c / T[p]:
            mup = c / T[p]
            muq = 0 
        elif mu[p] + dp < 0:
            mup = 0
            muq = c / T[q]
        else: #  0 <= mu[p] + dp <= c / T[p]のとき
            mup = mu[p] + dp
            muq = mu[q] - T[p] * T[q] * dp
    else: #T[p] == - T[q]のとき  
        if mu[p] + dp >= np.max([0, c / T[p]]):
            mup = mu[p] + dp
            muq = mu[q] - T[p] * T[q] * dp
        else: #mu[p] + dp < np.max([0, c / T[p]])のとき
            mup = np.max([0, c / T[p]])
            muq = c / T[q] + mup
    #4
    return np.array([mup, muq, p, q])
    
if __name__ == "__main__":
    #クラス1のデータ生成
    x1 = randn(N, 2) + np.array([4, 4])
    t1 = np.ones(N) 
    #クラス-1のデータ生成
    x2 = randn(N, 2) + np.array([-4, -4])
    t2 = - np.ones(N)
    
    #1と-1のデータ群を1つに統合
    X = np.append(x1, x2, axis = 0)    
    T = np.append(t1, t2, axis = 0)
    #双対問題に用いる変数muを生成(初期値1とする)
    mu = np.ones(2 * N)
    
    #パラメータw、thetaの初期値を生成
    w = X.T * (mu * T)
    w = np.array([np.sum(w[0]), np.sum(w[1])])
    theta = theta_f(X, T, mu)
    
    #SMO法の反復
    #回数は全データ数の半分で十分
    for i in range(N):
        a = SMO(X, T, theta, mu)
        
        #反復を打ち切るか判定
        if str(type(a)) == "<class 'bool'>":
            break
        
        #更新した値をmuに代入
        mu[np.int(a[2])] = a[0]
        mu[np.int(a[3])] = a[1]
        theta = theta_f(X, T, mu)  #thetaを更新
        
    #muの値を元にwを更新
    w = X.T * (mu * T)
    w = np.array([np.sum(w[0]), np.sum(w[1])])
    
    #異なるクラスからサポートベクトルを異なるクラスから1点ずつ選ぶ
    for i in range(len(mu)):
        if mu[i] != 0:
            if T[i] == 1:
                s = X[i]
            elif T[i] == -1:
                t = X[i]
    
    a = -w[0] / w[1] #識別面の傾き
    
    #このままだとthetaの値がずれてしまうので補正を行う.
    #点と直線の距離の公式|ax+by+c| / (a^2+b^2)^(1/2)が異なるサポートベクトルと識別面の間で
    #等しいことを利用してthetaの方程式を解く
    theta = (np.square(a * s[0] - s[1]) - np.square(a * t[0] - t[1])) \
                / (2 * ((s[1] - t[1]) - a * (s[0] - t[0])))
    
    #識別面
    px = np.linspace(-10, 10)
    py = a * px + theta
    
    plt.plot(x1[:, 0], x1[:, 1], ".")    
    plt.plot(x2[:, 0], x2[:, 1], ".")  
    
    #サポートベクトルを赤色にプロット
    for i in range(len(mu)):
        if mu[i] != 0:
            plt.plot(X[i][0], X[i][1], ".", color = "red") 
            
    plt.plot(px, py)
    plt.show()
    print("識別面の方程式:y = ", a, "x + (", theta, ")")
    print("サポートベクトル:")
    for i in range(len(mu)):
        if mu[i] != 0:
            print(X[i])