# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 18:39:46 2025

@author: 29535
"""

import numpy as np 
import GPy
import GPyOpt
import matplotlib.pyplot as plt


#Ru/Sr Flux Ratio
'''
# 你的实验数据（第一组）
Ru_flux_1 = np.array([0.20, 0.22, 0.29, 0.38, 0.40,0.50]).reshape(-1, 1)  # 第一组 Ru 通量 (Å/s)
RRR_1 = np.array([3.9, 5.4, 6.4, 13.5, 22.2, 32.0]).reshape(-1, 1)  # 第一组 RRR 值
'''

'''
#第二组数据)
Ru_flux_1 = np.array([0.20, 0.22, 0.29, 0.38, 0.40, 0.45, 0.61]).reshape(-1, 1)  # 第二组 Ru 通量 (Å/s)
RRR_1 = np.array([3.9, 5.4, 6.4, 13.5, 22.2, 0.0, 0.0]).reshape(-1, 1)  # 第二组 RRR 值
'''
'''
#第三组数据
Ru_flux_1 = np.array([0.20, 0.22, 0.29, 0.38, 0.40, 0.45, 0.61, 0.41, 0.42]).reshape(-1, 1)  # 第三组 Ru 通量 (Å/s)
RRR_1 = np.array(    [3.9, 5.4, 6.4, 13.5, 22.2, 0.0, 0.0, 25.0, 29.0]).reshape(-1, 1)  # 第三组 RRR 值
'''

#第四组数据
Ru_flux_1 = np.array([0.20, 0.22, 0.29, 0.38, 0.40, 0.45, 0.61, 0.41, 0.42, 0.33, 0.43]).reshape(-1, 1)  # 第四组 Ru 通量 (Å/s)
RRR_1 = np.array([3.9, 5.4, 6.4, 13.5, 22.2, 0.0, 0.0, 25.0, 29.0, 12.5, 13.2]).reshape(-1, 1)  # 第四组 RRR 值

# 选择数据
Ru_flux = Ru_flux_1  
RRR = RRR_1  

# 定义搜索空间
Ru_flux_bounds = (0.18, 0.61)  # Ru 通量的范围 (Å/s)
domain = [{'name': 'Ru_flux', 'type': 'continuous', 'domain': Ru_flux_bounds}]

# 训练 GPR（高斯过程回归）
kernel = GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=0.1)  
model = GPy.models.GPRegression(Ru_flux, RRR, kernel)

# 增加噪声方差，鼓励探索
model.Gaussian_noise.variance = 0.02**2  
model['.*Gaussian_noise.variance'].constrain_bounded(1e-5, 0.5)  

# **增加探索倾向** (增加 acquisition 的 kappa 参数)
bo = GPyOpt.methods.BayesianOptimization(f=None,  
                                         domain=domain,
                                         X=Ru_flux,
                                         Y=-RRR,  # 最大化 RRR
                                         model_type='GP',
                                         acquisition_type='EI',  
                                         acquisition_jitter=0.1,  # **增加探索**
                                         exact_feval=False,
                                         noise_var=0.02**2)

# 计算下一个 Ru_flux 采样点
next_point = bo.suggest_next_locations()  
next_Ru_flux = next_point[0, 0]  

# 可视化优化过程
Ru_flux_test = np.linspace(0.18, 0.61, 100).reshape(-1, 1)
mu, sigma = model.predict(Ru_flux_test)  

plt.figure(figsize=(8, 5))

# 蓝色实线：预测的 RRR 曲线
plt.plot(Ru_flux_test, mu, 'b-', label="predicted RRR")

# 红色虚线：预测的 RRR ± σ 区间
plt.fill_between(Ru_flux_test.flatten(), (mu - 5 * sigma).flatten(),
                 (mu + 5 * sigma).flatten(), color='r', alpha=0.2, label="predicted RRR ± σ")

# 黑色实心圆：实验数据点
plt.scatter(Ru_flux, RRR, color='k', label="experimental RRR", zorder=5)

# 紫色虚线：建议的下一个采样点
plt.axvline(x=next_Ru_flux, color='purple', linestyle='--', label=f"next sampling point ({next_Ru_flux:.3f} Å/s)")

# 设置纵坐标最小值为 -1，避免负值
plt.ylim(bottom=-1)

# 绘图标签
plt.xlabel("Ru flux (Å/s)")
plt.ylabel("RRR")
plt.title("optimization of Ru flux")
plt.legend()
plt.show()

# 输出优化结果
print("优化完成！")
print(f"建议的下一个 Ru 通量: {next_Ru_flux:.3f} Å/s")
