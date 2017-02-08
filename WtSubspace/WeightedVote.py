import numpy as np

class WV():
  def __init__(self,size,times,X,y,base_learner,seed1,seed2,weight_schema):
    self.size = size
    self.times = times
    self.X = X
    self.y = y
    self.classifier = base_learner
    self.random_seed1 = seed1
    self.random_seed2 = seed2
    self.weight_schema = weight_schema

  def fit(self):
    len_features = np.array(len(list(self.X)))
    np.random.seed(self.random_seed1)
    sub_index = [np.random.choice(len_features, self.size, replace=False) for x in range(self.times)]
    score_list = []
    preds_ = []


    for i in range(self.times):
        dat = self.X.iloc[:, sub_index[i]]

        if self.classifier == 'svm':
            from sklearn.svm import SVC
            model = SVC()
            model.fit(dat, self.y)
        elif self.classifier == 'tree':
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(random_state=1, max_depth = 3)
            model.fit(dat, self.y)

        score = model.score(dat, self.y)
        if score <= 0.5:
            score = 0.5
        score_list.append(score)
        pred = model.predict(dat)
        preds_.append(pred)

    error = 1 - np.array(score_list)
    unnorm_weight = np.log((1 - error) / error)
    
    if self.weight_schema == 'independent':
        weight = unnorm_weight / unnorm_weight.sum()
    elif self.weight_schema == 'equal':
        weight = np.ones(len(preds_))/len(preds_)

    agg_pred = []
    np.random.seed(self.random_seed2)
    for j in range(len(self.X)):
        k = np.random.choice(np.array(preds_).T[j], p=weight)
        agg_pred.append(k)
    self.agg_pred = agg_pred

    return self

  def score(self,z):

    return len([x for x in zip(self.agg_pred, z) if x[0] == x[1]])/1. / len(z)



