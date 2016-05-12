#coding:utf-8
#
import numpy as np
import csv as csv
dir=r'E:\kaggle\titanic\data\train.csv'
csv_file=csv.reader(open(dir))
header=csv_file.next() #读取头文件了
data1=[i for i in csv_file]
data=np.array(data1)  #转成np格式

women_only_stats=data[:,4]=='female'
men_only_stats=data[:,4]!='female'
#将男女生存下来的情况分开
women_onboard=data[women_only_stats,1].astype(np.float)
men_onboard=data[men_only_stats,1].astype(np.float)

pro_women_survived=np.sum(women_onboard)/np.size(women_onboard)
pro_men_survived=np.sum(men_onboard)/np.size(men_onboard)
print 'Proportion of women who survived is %s' % pro_women_survived
print 'Proportion of men who survived is %s' % pro_men_survived


#session 2
#最简单的预测test数据集中 认为男的survive=0 女的survive=1
#打开test测试集
file2=r'E:\kaggle\titanic\data\test.csv'
predict_file=r'E:\kaggle\titanic\data\genderpredict.csv'
predict=open(predict_file,'wb')
predict_object=csv.writer(predict)

test_data=csv.reader(open(file2))
header2=test_data.next()
predict_object.writerow(['PassengerId','Survived'])
for row in test_data:
    if row[3]=='female' :
        predict_object.writerow([row[0],'1'])
    else:
        predict_object.writerow([row[0],'0'])
predict.close()



        
        
