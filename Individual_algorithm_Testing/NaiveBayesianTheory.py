import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
sns.set(color_codes=True)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer,confusion_matrix
import numpy as np
np.random.seed(97)



diab=pd.read_csv("diabetes.csv")
#print(diab.head())
#print(diab.isnull().values.any())
#print(diab.describe())
print(((diab.Pregnancies == 0).sum(),
       (diab.Glucose == 0).sum(),
       (diab.BloodPressure == 0).sum(),
       (diab.SkinThickness==0).sum(),
       (diab.Insulin==0).sum(),
       (diab.BMI==0).sum(),
       (diab.DiabetesPedigreeFunction==0).sum(),
       (diab.Age==0).sum()))

drop_Gluc=diab.index[diab.Glucose == 0].tolist()
drop_BP=diab.index[diab.BloodPressure == 0].tolist()
drop_Skin=diab.index[diab.SkinThickness==0].tolist()
drop_Ins=diab.index[diab.Insulin==0].tolist()
drop_BMI=diab.index[diab.BMI==0].tolist()
c=drop_Gluc+drop_BP+drop_Skin+drop_Ins+drop_BMI
dia=diab.drop(diab.index[c])
#print(dia.describe())
#Data that we can able to see

dia1 = dia[dia.Outcome==1]
dia0=dia[dia.Outcome==0]
#print(dia1)
#print(dia0)

#percentage of the Outcome
Out0 = len(dia[dia.Outcome==1])
Out1 = len(dia[dia.Outcome==0])
Total = Out0+Out1
PC_of_1 = Out1*100/Total
PC_of_0 = Out0*100/Total
#print(PC_of_0,PC_of_1)

#Algorithm Procedure starts
cols=["Glucose","BloodPressure","Insulin","BMI","Age"]
x=dia[cols]
y=dia.Outcome

#defining Function
def tn(y_true,y_pred):return confusion_matrix(y_true,y_pred)[0,0]
def fp(y_true,y_pred):return confusion_matrix(y_true,y_pred)[0,1]
def fn(y_true,y_pred):return confusion_matrix(y_true,y_pred)[1,0]
def tp(y_true,y_pred):return confusion_matrix(y_true,y_pred)[1,1]


#searched Out Feature
scoring = {'accuracy':make_scorer(accuracy_score),'prec':'precision'}
scoring = {'tp':make_scorer(tp),'tn':make_scorer(tn),'fp':make_scorer(fp),'fn':make_scorer(fn)}


#spliting Train set and Test setb
train_X,test_X,train_Y,test_Y=train_test_split(x,y,test_size=0.2)
clf=GaussianNB()
clf.fit(train_X,train_Y)
logreg=GaussianNB().fit(train_X,train_Y)
print("Training set score : {0}".format(logreg.score(train_X,train_Y)))
print("Testing  set score : {0}".format(logreg.score(test_X,test_Y)))
y_pred = clf.predict(test_X)
ac = accuracy_score(test_Y,y_pred)
rc = roc_auc_score(test_Y,y_pred)

#Result Execution step
result = cross_validate(clf,train_X,train_Y,scoring=scoring,cv=10)
print("Accuracy {0} ROC {1}".format(ac,rc))