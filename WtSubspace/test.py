from WeightedVote import WV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd



X = pd.read_table('Datasets/wheat.data',delimiter=',')

X.dropna(axis=0,inplace=True)
y = X.loc[:,'wheat_type']
y = y.map({'canadian': 0, 'rosa': 1, 'kama': 2})
X.drop(axis=1,labels=['id','wheat_type'],inplace=True)


model_svm1 = WV(size=5,times=200,X=X,y=y,base_learner='svm',seed1=1,seed2=123,weight_schema='independent')
model_svm1.fit()
model_svm2 = WV(size=5,times=200,X=X,y=y,base_learner='svm',seed1=1,seed2=123,weight_schema='equal')
model_svm2.fit()

model_tree1 = WV(size=5,times=200,X=X,y=y,base_learner='tree',seed1=1,seed2=123,weight_schema='independent')
model_tree1.fit()
model_tree2 = WV(size=5,times=200,X=X,y=y,base_learner='tree',seed1=1,seed2=123,weight_schema='equal')
model_tree2.fit()

print ("SVM base learner accuaracy with independent weight: {0:.3f}\n".format(model_svm1.score(y)))
print ("SVM base learner accuaracy with equal weight: {0:.3f}\n".format(model_svm2.score(y)))

print ("Tree base learner accuaracy with independent weight: {0:.3f}\n".format(model_tree1.score(y)))
print ("Tree base learner accuaracy with equal weight: {0:.3f}\n".format(model_tree2.score(y)))



### compare with SVM and RF
model_svm_alone = SVC()
model_svm_alone.fit(X,y)
print ("SVM only accuracy: {0:.3f}\n".format(model_svm_alone.score(X,y)))


model_rf = RandomForestClassifier(random_state=123,max_features='sqrt',max_depth=3)
model_rf.fit(X,y)
print ("Random Forest accuracy: {0:.3f}\n".format(model_rf.score(X,y)))


