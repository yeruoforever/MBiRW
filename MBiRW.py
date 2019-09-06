import numpy as np
from numpy import ndarray, sqrt

# Laplace 正则化


def Laplacian_normalization(M: ndarray):
    for i in range(M.shape[0]):
        M[i, i] = sum(M[i, :])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            M[i, j] /= sqrt(M[i, i]*M[j, j])
    return M

# MBiRW算法


def MBiRW(simR: ndarray, simD: ndarray, A: ndarray, a: float, l: int, r: int):
    MR = Laplacian_normalization(simR)
    MD = Laplacian_normalization(simD)
    A = A/A.sum()
    RD = A
    for i in range(max(l, r)):
        dflag = 1
        rflag = 1
        if i <= l:
            Rr = a*np.matmul(RD, MR)+(1-a)*A
            rflag = 1
        if i <= r:
            Rd = a*np.matmul(MD, RD)+(1-a)*A
            dflag = 1
        RD = (rflag * Rr + dflag * Rd) / (rflag + dflag)
    return RD


# 读取数据
simR = np.loadtxt("./Datasets/DrugSimMat")
simD = np.loadtxt("./Datasets/DiseaseSimMat")
A = np.loadtxt("./Datasets/DiDrAMat")
# 设置参数
l = 2
r = 2
a = 0.3
# 执行
RD = MBiRW(simR, simD, A, a, l, r)
# 保持
np.savetxt(str(a) + "PyRD.Mat", RD)
# 模型定义


