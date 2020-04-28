import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#下载数据
iris_data = load_iris()
X = iris_data['data']
y = iris_data['target']

#数据标准化处理
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

#计算协方差矩阵
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('协方差矩阵 \n%s' %cov_mat)
'''
协方差矩阵 
[[ 1.00671141 -0.11010327  0.87760486  0.82344326]
 [-0.11010327  1.00671141 -0.42333835 -0.358937  ]
 [ 0.87760486 -0.42333835  1.00671141  0.96921855]
 [ 0.82344326 -0.358937    0.96921855  1.00671141]]
 '''
 #也可以直接使用numpy封装函数直接求解
print('NumPy 计算协方差矩阵: \n%s' %np.cov(X_std.T))
'''
NumPy 计算协方差矩阵: 
[[ 1.00671141 -0.11010327  0.87760486  0.82344326]
 [-0.11010327  1.00671141 -0.42333835 -0.358937  ]
 [ 0.87760486 -0.42333835  1.00671141  0.96921855]
 [ 0.82344326 -0.358937    0.96921855  1.00671141]]
 '''
 #计算特征向量与特征值
cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('特征向量 \n%s' %eig_vecs)
print('\n特征值 \n%s' %eig_vals)
'''
特征向量 
[[ 0.52237162 -0.37231836 -0.72101681  0.26199559]
 [-0.26335492 -0.92555649  0.24203288 -0.12413481]
 [ 0.58125401 -0.02109478  0.14089226 -0.80115427]
 [ 0.56561105 -0.06541577  0.6338014   0.52354627]]

特征值 
[2.93035378 0.92740362 0.14834223 0.02074601]
'''
#取前两个特征值
matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                      eig_pairs[1][1].reshape(4,1)))

print('Matrix W:\n', matrix_w)
'''
Matrix W:
 [[ 0.52237162 -0.37231836]
 [-0.26335492 -0.92555649]
 [ 0.58125401 -0.02109478]
 [ 0.56561105 -0.06541577]]
 '''
 #得到新的特征
 Y = X_std.dot(matrix_w)
 
 '''
 #原始数据分布，取其中两个维度
plt.figure(figsize=(6, 4))
for lab, col in zip((0, 1, 2),
                        ('blue', 'red', 'green')):
     plt.scatter(X[y==lab, 0],
                X[y==lab, 1],
                label=lab,
                c=col)
plt.xlabel('sepal_len')
plt.ylabel('sepal_wid')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#转换后数据分布
plt.figure(figsize=(6, 4))
for lab, col in zip((0, 1, 2),
                        ('blue', 'red', 'green')):
     plt.scatter(Y[y==lab, 0],
                Y[y==lab, 1],
                label=lab,
                c=col)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='lower center')
plt.tight_layout()
plt.show()
 '''
 
 #建模测试
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import accuracy_score

X_train,X_test,y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
predictions=classifier.predict(X_test)

acc = accuracy_score(y_test, predictions)
print(acc)
#0.8888888888888888

X_train_n,X_test_n,y_train_n, y_test_n = train_test_split(Y,y, test_size=0.3, random_state=0)
classifier.fit(X_train_n,y_train_n)
predictions_n=classifier.predict(X_test_n)

acc_n = accuracy_score(y_test_n, predictions_n)
print(acc_n)
#0.8666666666666667

 
 







 
 
 
 
 
 
 




