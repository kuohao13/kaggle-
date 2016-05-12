#coding:utf-8
'''
Created on 2016年5月12日
@author: kuohao
'''

#data:泰坦尼克得到数据
#使用性别，class，船票 三个特征来判断 是否生存
#方法：建立一个（性别）2X3（类别）X4船票 的概率矩阵， 判断落在这一类的生存下来的概率

import numpy as np
import csv as csv 
#打开文件
filedir=open(r'E:\kaggle\titanic\data\train.csv')
testdata_obj=csv.reader(filedir)
#跳过头部 列名
header=testdata_obj.next()
#列名  ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch',
#  'Ticket','Fare','Cabin','Embarked']
#获取列表
testdata=[i for i in testdata_obj]
#将列表转为np array
testdata1=np.array(testdata)
#testdata1[0,:]
#array(['1', '0', '3', 'Braund, Mr. Owen Harris', 'male', '22', '1', '0',
#       'A/5 21171', '7.25', '', 'S']

#获得Pclass的类别与数量
Pclass_num=np.unique(testdata1[:,2].astype(np.int32))
number_of_Pclass=len(Pclass_num)
#为票价进行分组 testda[:,9] Fare部分 
#分成[0-9],[10-19],[20-29],[30-39] 四个部分  对于大于39以上的数据归为[30-39]内
fare_ceiling=40
tick_Fare=testdata1[testdata1[:,9].astype(np.float)>=fare_ceiling,9]=fare_ceiling-1
fare_bracker_size=10
number_of_price_braceker=fare_ceiling/fare_bracker_size

#初始化幸存下来的这三个类别的人的概率矩阵
survival_array=np.zeros((2,number_of_Pclass,number_of_price_braceker)) #就是2x3x4的一个矩阵

for i in Pclass_num:
    for j in xrange(number_of_price_braceker):
        #得到women的满足条件的 survived列的数据了
        women_stats=testdata1[(testdata1[:,4]=='female') &
                             (testdata1[:,2].astype(np.int32)==i) &
                             (testdata1[:,9].astype(np.float)<(j+1)*fare_bracker_size) &
                             (testdata1[:,9].astype(np.float)>=j*fare_bracker_size) ,1]
        men_stats=testdata1[(testdata1[:,4]=='male') &
                             (testdata1[:,2].astype(np.int32)==i) &
                             (testdata1[:,9].astype(np.float)<(j+1)*fare_bracker_size) &
                             (testdata1[:,9].astype(np.float)>=j*fare_bracker_size) ,1]
        survival_array[0,i-1,j]=np.mean(women_stats.astype(np.float))
        survival_array[1,i-1,j]=np.mean(men_stats.astype(np.float))
print survival_array
#-------------------概率矩阵如下--------------------------
'''array([[[        nan,         nan,  0.83333333,  0.97727273],
        [        nan,  0.91428571,  0.9       ,  1.        ],
        [ 0.59375   ,  0.58139535,  0.33333333,  0.125     ]],

       [[ 0.        ,         nan,  0.4       ,  0.38372093],
        [ 0.        ,  0.15873016,  0.16      ,  0.21428571],
        [ 0.11153846,  0.23684211,  0.125     ,  0.24      ]]])
'''
#-------------------------------------------------------
#可以说使用survival_array！=survival_array 获取nan 的索引
#将nan部分赋值为0
survival_array[survival_array!=survival_array]=0.

#认为概率矩阵中 大于.5表示幸存下来 赋值为1
survival_array[survival_array>=.5]=1
survival_array[survival_array<.5]=0


#将训练的模型 用到测试集中
test_file = open(r'E:\kaggle\titanic\data\test.csv', 'rb')
test_file_object = csv.reader(test_file)
test_header = test_file_object.next()
predictions_file = open(r'E:\kaggle\titanic\data\genderclassmodel.csv', "wb")
p = csv.writer(predictions_file)
p.writerow(["PassengerId", "Survived"])

#需要对Fare票价的数据进行处理  因为在model中  Fare被分成了4类  即j 0,1,2,3,所以需要将test集中处理下
for row in test_file_object:
    for j in xrange(number_of_price_braceker):
        try:
            row[8]=float(row[8])
        except:
            bin_fare=3-float(row[1]) #如果测试集中没有fare数据用class代替
            break
        if row[8]>fare_ceiling:
            bin_fare=number_of_price_braceker-1 #超过最大是，则用[30-39] j=3表示
            break
        if ((row[8]>=j*fare_bracker_size)            
            and (row[8]<(j+1)*fare_bracker_size)):
            bin_fare=j
            break
    if row[3]=='female':
        p.writerow([row[0],'%d' %int(survival_array[0,float(row[1])-1,bin_fare])])
    else:
        p.writerow([row[0],'%d' %int(survival_array[1,float(row[1])-1,bin_fare])])       
        
test_file.close()
predictions_file.close()        
         
        

