from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
data_tr = pd.read_csv(r'./mask.csv')   # 导入数据

model = MLPRegressor(hidden_layer_sizes=(10,), random_state=10,learning_rate_init=0.1)  # BP神经网络回归模型
model.fit(data_tr.iloc[:,:4],data_tr.iloc[:,4])  # 训练模型
while 1==1:
    a = int(input("第一个成绩："))
    b = int(input("第二个成绩："))
    c = int(input("第三个成绩："))
    d = int(input("第四个成绩："))
    pre = model.predict([[a,b,c,d]])

    print(pre)