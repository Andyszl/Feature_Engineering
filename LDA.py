from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#下载数据
iris_data = load_iris()
X = iris_data['data']
y = iris_data['target']

import numpy as np
#设置小数点的位数
np.set_printoptions(precision=4) 
#这里会保存所有的均值
mean_vectors = []
# 要计算3个类别
for cl in range(0,3):
    # 求当前类别各个特征均值
    mean_vectors.append(np.mean(X[y==cl], axis=0))
    print('均值类别 %s: %s\n' %(cl, mean_vectors[cl]))
'''
均值类别 0: [5.006 3.418 1.464 0.244]
均值类别 1: [5.936 2.77  4.26  1.326]
均值类别 2: [6.588 2.974 5.552 2.026]
'''

# 原始数据中有4个特征
S_W = np.zeros((4,4))
# 要考虑不同类别，自己算自己的
for cl,mv in zip(range(0,4), mean_vectors):
    class_sc_mat = np.zeros((4,4))    
    # 选中属于当前类别的数据
    for row in X[y == cl]:
        # 这里相当于对各个特征分别进行计算，用矩阵的形式
        row, mv = row.reshape(4,1), mv.reshape(4,1)
        # 跟公式一样
        class_sc_mat += (row-mv).dot((row-mv).T)
    S_W += class_sc_mat                             
print('类内散布矩阵:\n', S_W)    

'''
类内散布矩阵:
 [[38.9562 13.683  24.614   5.6556]
 [13.683  17.035   8.12    4.9132]
 [24.614   8.12   27.22    6.2536]
 [ 5.6556  4.9132  6.2536  6.1756]]
 '''

overall_mean = np.mean(X, axis=0)
# 构建类间散布矩阵
S_B = np.zeros((4,4))
# 对各个类别进行计算
for i,mean_vec in enumerate(mean_vectors):  
    #当前类别的样本数
    n = X[y==i,:].shape[0]
    mean_vec = mean_vec.reshape(4,1) 
    overall_mean = overall_mean.reshape(4,1) 
    # 如上述公式进行计算
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

print('类间散布矩阵:\n', S_B)

'''
类间散布矩阵:
 [[ 63.2121 -19.534  165.1647  71.3631]
 [-19.534   10.9776 -56.0552 -22.4924]
 [165.1647 -56.0552 436.6437 186.9081]
 [ 71.3631 -22.4924 186.9081  80.6041]]
 '''
#求解矩阵特征值，特征向量
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
# 拿到每一个特征值和其所对应的特征向量
for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:,i].reshape(4,1)   
    print('\n特征向量 {}: \n{}'.format(i+1, eigvec_sc.real))
    print('特征值 {:}: {:.2e}'.format(i+1, eig_vals[i].real))
    
    
'''
特征向量 1: 
[[ 0.2049]
 [ 0.3871]
 [-0.5465]
 [-0.7138]]
特征值 1: 3.23e+01
特征向量 2: 
[[-0.009 ]
 [-0.589 ]
 [ 0.2543]
 [-0.767 ]]
特征值 2: 2.78e-01
特征向量 3: 
[[ 0.2628]
 [-0.3635]
 [-0.4127]
 [ 0.6229]]
特征值 3: -2.17e-15
特征向量 4: 
[[ 0.2628]
 [-0.3635]
 [-0.4127]
 [ 0.6229]]
特征值 4: -2.17e-15
'''

#特征值和特征向量配对
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# 按特征值大小进行排序
eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

print('特征值排序结果:\n')
for i in eig_pairs:
    print(i[0])
    
'''
特征值排序结果:
32.27195779972981
0.27756686384004514
5.422091066853206e-15
5.422091066853206e-15
'''

print('特征值占总体百分比:\n')
eigv_sum = sum(eig_vals)
for i,j in enumerate(eig_pairs):
    print('特征值 {0:}: {1:.2%}'.format(i+1, (j[0]/eigv_sum).real))
'''
特征值占总体百分比:
特征值 1: 99.15%
特征值 2: 0.85%
特征值 3: 0.00%
特征值 4: 0.00%
'''

#选取前两个
W = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1)))
print('矩阵W:\n', W.real)
'''
矩阵W:
 [[ 0.2049 -0.009 ]
 [ 0.3871 -0.589 ]
 [-0.5465  0.2543]
 [-0.7138 -0.767 ]]
 '''
 
 # 执行降维操作
X_lda = X.dot(W)
X_lda.shape

(150, 2)

# 可视化展示
label_dict = {i:label for i,label in zip(
                range(0,3),
                  ('Setosa',
                  'Versicolor',
                  'Virginica'
                 ))}
from matplotlib import pyplot as plt

def plot_step_lda():

    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(0,3),('^', 's', 'o'),('blue', 'red', 'green')):

        plt.scatter(x=X_lda[:,0].real[y == label],
                y=X_lda[:,1].real[y == label],
                marker=marker,
                color=color,
                alpha=0.5,
                label=label_dict[label]
                )

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('LDA on iris')

    # 把边边角角隐藏起来
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # 为了看的清晰些，尽量简洁
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

    plt.grid()
    plt.tight_layout
    plt.show()

plot_step_lda()

#结果测试
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import accuracy_score


#原始数据
X_train,X_test,y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
predictions=classifier.predict(X_test)

acc = accuracy_score(y_test, predictions)
print(acc)
#0.8888888888888888

#降维后
X_train_n,X_test_n,y_train_n, y_test_n = train_test_split(X_lda,y, test_size=0.3, random_state=0)

classifier.fit(X_train_n,y_train_n)
predictions_n=classifier.predict(X_test_n)

acc_n = accuracy_score(y_test_n, predictions_n)
print(acc_n)
#0.8444444444444444
