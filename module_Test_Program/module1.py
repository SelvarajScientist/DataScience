import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
sns.set(color_codes=True)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
#DATASET INITIALIZATION
diab=pd.read_csv('diabetes.csv')

#OCCURENNCE OF ZEROS
#print(diab.head())
#print(diab.isnull().values.any())
#print(diab.describe())
#print(((diab.Pregnancies == 0).sum(),
#       (diab.Glucose == 0).sum(),
#      (diab.BloodPressure == 0).sum(),
#      (diab.SkinThickness==0).sum(),
#       (diab.Insulin==0).sum(),
#       (diab.BMI==0).sum(),
#       (diab.DiabetesPedigreeFunction==0).sum(),
#       (diab.Age==0).sum()))


#DROP THE ZERO OCCURENECE VALUES
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

#Segregation of dataset for the training
cols=["Glucose","BloodPressure","Insulin","BMI","Age"]
x=dia[cols]
y=dia.Outcome
train_X,test_X,train_Y,test_Y=train_test_split(x,y,test_size=0.2)

#Manually entering the values check
test_X=[[84,82,125,38.2,23]]

#Algorithm procedure starts for Gradient Booster Algorithm
GradientBooster=GradientBoostingClassifier(n_estimators=50,learning_rate=0.2)
GradientBooster =GradientBooster.fit(train_X,train_Y)
pred_GradientBooster = GradientBooster.predict(test_X)
pred_GradientBooster_Prob = GradientBooster.predict_proba(test_X)

#Algorithm procedure starts for Naive_Bayes
Navie_bayes=GaussianNB()
Navie_bayes=Navie_bayes.fit(train_X,train_Y)
pred_Naive = Navie_bayes.predict(test_X)
pred_Naive_Prob = Navie_bayes.predict_proba(test_X)

#Algorithm procedure starts for SVM
Supporvectormachine=SVC(kernel='linear',probability=True)
Supporvectormachine=Supporvectormachine.fit(train_X,train_Y)
pred_SVM = Supporvectormachine.predict(test_X)
pred_SVM_Prob = Supporvectormachine.predict_proba(test_X)
#pred_SVM_Prob = Supporvectormachine.predict_proba(test_X)


#printing of values
print('INPUTED LEVEL OF GLUCOSE',test_X[0][0])
print('INPUTED LEVEL OF BLOODPRESSURE',test_X[0][1])
print('INPUTED LEVEL OF INSULIN',test_X[0][2])
print('INPUTED LEVEL OF BMI',test_X[0][3])
print('INPUTED LEVEL OF AGE',test_X[0][4])
print('STATUS PREDICTIED BY GRADIENT BOOSTER ALGORITHM IS:',pred_GradientBooster)
print('STATUS PREDICTED BY NAIVE BAYESIAN ALGORITHM IS:',pred_Naive)
print('STATUS PREDICTED BY SUPPORT VECTOR MACHINE IS:',pred_SVM)
print('PREDICTED OCCURENCE PERCENTAGE OF PERSON HAVING DIABETES USING GRADIENTBOOSTER IS ',int(round((pred_GradientBooster_Prob[0][1])*100)),'%')
print('PREDICTED OCCURENCE PERCENTAGE OF PERSON NOT HAVING DIABETES USIN GRADIENTBOOSTER IS ',int(round(pred_GradientBooster_Prob[0][0]*100)),'%')
print('PREDICTED OCCURENCE PERCENTAGE OF PERSON HAVING DIABETES USING NAIVESBAYES IS ',int(round(pred_Naive_Prob[0][1]*100)),'%')
print('PREDICTED OCCURENCE PERCENTAGE OF PERSON NOT HAVING DIABETES USING NAIVESBAYES IS ',int(round(pred_Naive_Prob[0][0]*100)),'%')
print('PREDICTED OCCURENCE PERCENTAGE OF PERSON  HAVING DIABETES USING SUPPORTVECTORMACHINE IS ',int(round(pred_SVM_Prob[0][1]*100)),'%')
print('PREDICTED OCCURENCE PERCENTAGE OF PERSON NOT HAVING DIABETES USING SUPPORTVECTORMACHINE IS ',int(round(pred_SVM_Prob[0][0]*100)),'%')


#vecortclassifier
Overall_Mean=VotingClassifier(estimators=[
                                            ('GradientBooster',GradientBooster),
                                            ('Naive_bayes',Navie_bayes),
                                            ('SupportVectorMachine',Supporvectormachine)],
                              weights=[2,2,1],
                              flatten_transform=True,
                              voting='soft').fit(train_X,train_Y)
Overall_fit=Overall_Mean.predict_proba(test_X)
print('OVERALL PREDICTED PERCENTAGE OF PERSON HAVING DIABETES USING VECTOR CLASSIFIER IS',int(round(Overall_fit[0][1]*100)),'%')
print('OVERALL PREDICTED PERCENTAGE OF PEARSON NOT HAVING DIABETES USING VECTOR CLASSIFIER IS',int(round(Overall_fit[0][0]*100)),'%')

#pickle for importing the file
import pickle
filename='H:\lifecoaching_test\pickle'
outfile = open(filename,'wb')
pickle.dump(Overall_Mean,outfile)
outfile.close()

infile = open(filename,'rb')
new_dict = pickle.load(infile)
result = new_dict.predict(test_X)
print(result)


