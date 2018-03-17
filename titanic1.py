import pandas as pd

train = pd.read_csv('C:/python36/train.csv')
test = pd.read_csv('C:/python36/test.csv')

train.head()
test.head()

train.shape
test.shape

train.info()
test.info()

train.isnull().sum()
test.isnull().sum()

import matplotlib.pyplot as plt
import seaborn as sns

def bar_chart(feature) :
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))

#feature별로 알아보기    
bar_chart('Sex')
bar_chart('Pclass')
bar_chart('SibSp')
bar_chart('Parch')
bar_chart('Embarked')

##Feature engineering

#name을 title로
train_test_data = [train, test]

for dataset in train_test_data :
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=False)

train['Title'].value_counts()
test['Title'].value_counts()

##title을 classify하기
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)

bar_chart('Title')

##불필요한 name 제거
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

##Sex를 Classify하기
sex_mapping = {'male' : 0, 'female' : 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(title_mapping)
    
##Age의 Nan제거
    train.head(100)

#잃어버린 Age를 Title의 평균나이로 채우기 (Mr,Mrs,Miss, Others)
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)

facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0, train["Age"].max()))
facet.add_legend()
plt.show()

facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0, train["Age"].max()))
facet.add_legend()
plt.xlim(0,20)

facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0, train["Age"].max()))
facet.add_legend()
plt.xlim(20,30)

facet = sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0, train["Age"].max()))
facet.add_legend()
plt.xlim(40,60)

#나이군 끼리 묶기
'''
요소의 백터 맵핑:
child:0
young:1
adult:2
mid-age:3
senior:4
'''
for dataset in train_test_data:
    dataset.loc[dataset['Age'] <=16, "Age"] =0,
    dataset. loc[(dataset['Age'] >16) & (dataset['Age'] <=26), 'Age'] =1,
    dataset. loc[(dataset['Age'] >26) & (dataset['Age'] <=36), 'Age'] =2,
    dataset. loc[(dataset['Age'] >36) & (dataset['Age'] <=62), 'Age'] =3,
    dataset.loc[dataset['Age'] >62, "Age"] =4

train.head()
bar_chart("Age")

##Embarked 요소
#누락된 요소 채우기(Pclass값 참조)
Pclass1 = train[train["Pclass"]==1]['Embarked'].value_counts()
Pclass2 = train[train["Pclass"]==2]['Embarked'].value_counts()
Pclass3 = train[train["Pclass"]==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ('1st class', '2nd class','3rd class')
df.plot(kind='bar', stacked=True, figsize=(10,5))
'''
S값이 많이나와서 Embark를 S로 해줘도 무방
'''

for dataset in train_test_data :
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

train.head()

# 매핑하기
embarked_mapping = {'S' : 0, "C" : 1, "O" : 2}
for dataset in train_test_data :
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

##Fare 요소
#누락요소 채우기(Pclass참조)
train["Fare"].fillna(train.groupby('Pclass')['Fare'].transform("median"), inplace=True)
test["Fare"].fillna(train.groupby('Pclass')['Fare'].transform("median"), inplace=True)

#그래프 그려서 확인하기
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
 
plt.show()

facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
plt.Xlim(0,20)

facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
plt.Xlim(0,30)

for dataset in train_test_data :
    dataset. loc[dataset['Fare'] <= 17, 'Fare'] =0,
    dataset. loc[(dataset['Fare'] > 17) &(dataset['Fare'] <=30), 'Fare'] =1,
    dataset. loc[(dataset['Fare'] > 30) &(dataset['Fare'] <=100), 'Fare'] =2,
    dataset. loc[dataset['Fare'] > 100, 'Fare'] = 3

train.head()

##Cabin요소
train.Cabin.value_counts()

for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]

Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df =pd.DataFrame([Pclass1 , Pclass2 , Pclass3])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind='bar', stacked=True, figsize=(10,5))
 
'''
feature scaling은 컴퓨터는 숫자값의 차이만큼 차이를 인식 => 차이가 크지 않은것은
벡터값으로 줄 때 차이를 줄여야 함 (A~D의 의 등급이 같다면 소숫점으로 구분)
'''
#매핑하기
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

#Pclass의 중간값을 Cabin의 누락값에 넣어주기
train['Cabin'].fillna(train.groupby("Pclass")['Cabin'].transform('median'), inplace=True)
test['Cabin'].fillna(train.groupby("Pclass")['Cabin'].transform('median'), inplace=True)

##FamillySize요소 만들어서 가족의 크기 벡터화 
#FamillySize요소를 Sibsp, Parch를 합셔서 만들기
train['FamilySize'] = train['SibSp'] + train["Parch"] + 1
test['FamilySize'] = test['SibSp'] + test["Parch"] + 1

facet= sns.FacetGrid(train, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'FamilySize', shade= True)
facet.set(xlim=(0,train['FamilySize'].max()))
facet.add_legend()
plt.xlim(0)

#매핑하기
family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)

#불필요한 요소 drop하기
features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)


train_data = train.drop('Survived', axis=1)
target = train['Survived']

train_data.shape, target.shape

###모델링하기
#Classifier 모듈을 import
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np

#cross validation을 위해
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

##kNN
clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
#kNN score
round(pnp.mean(score)*100,2)

##Decision Tree
clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
#Decision tree 점수
round(np.mean(score)*100, 2)

##Random Forest
clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
#Random Forest 점수
round(np.mean(score)*100, 2)

##Naive Bayes
clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
#Naive Bayes 점수
round(np.mean(score)*100, 2)

##SVM
clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
#SVM점수
round(np.mean(score)*100,2)


###########Testing################
clf = SVC()
clf.fit(train_data, target)

test_data = test.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)

'''
인용
https://github.com/minsuk-heo/kaggle-titanic/blob/master/titanic-solution.ipynb
'''

