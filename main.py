import numpy as np
import pandas as pd
from scipy.optimize import minimize

# データを配列として定義します
data = np.array([
    [0.0285699, 0.029169621, 0.038634213, 0.058941792, 0.004716124, 0.026023502],
    [-0.033859569, -0.108917004, -0.125681544, -0.023862959, -0.188232716, -0.050660311],
    [0.183257394, 0.17112084, 0.164006776, 0.240675383, 0.217476173, 0.182905223],
    [0.548508781, 0.54550215, 0.637857369, 0.521989789, 0.537788421, 0.716824581],
    [0.151685386, 0.151581039, 0.173957586, 0.144985207, 0.115981338, 0.096178933],
    [0.144553797, 0.08176432, 0.087807144, 0.131862775, 0.107830596, 0.121097041],
    [0.034716838, 0.060462391, 0.111436685, 0.059859906, 0.003080072, -0.034813505],
    [0.276458377, 0.225468275, 0.350787305, 0.276067941, 0.215908909, 0.319686178],
    [-0.208513566, -0.167220801, -0.197904335, -0.140970948, -0.143898871, -0.307261882],
    [0.223685309, 0.228404934, 0.234604113, 0.123583788, 0.173008691, 0.156956026],
    [0.123170903, -0.05950225, -0.057795451, -0.096568429, 0.102332428, 0.161247106],
    [0.046568918, 0.19697369, 0.230825562, 0.269572626, 0.137809117, 0.017638768],
    [-0.024060451, 0.199541287, 0.171929564, 0.173264438, 0.000967423, 0.015853893],
    [0.294335059, 0.360656835, 0.25489869, 0.377252954, 0.305017088, 0.28313764]
])

# 目的関数の定義
def objective(weights):
    weights = np.array(weights)
    annual_returns = np.prod(1 + data * weights, axis=1)
    total_return = np.prod(annual_returns)
    return -total_return  # 最適化は最小化なので、マイナスにして最大化を目指す

# 制約条件
def constraint_sum_of_weights(weights):
    return np.sum(weights) - 1

# 初期重み
initial_weights = np.ones(data.shape[1]) / data.shape[1]

# 制約と境界
constraints = ({'type': 'eq', 'fun': constraint_sum_of_weights})
bounds = [(0, 1)] * data.shape[1]

# 最適化の実行
result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)

# 結果を確認
if result.success:
    optimal_weights = result.x
    annual_returns = np.prod(1 + data * optimal_weights, axis=1)
    total_return = np.prod(annual_returns) - 1
    total_return_percentage = total_return * 100
    print(f"Optimal weights: {optimal_weights}")
    print(f"Total return percentage: {total_return_percentage}%")
else:
    print("Optimization failed.")
    print(result)

result
