import numpy as np
import pandas as pd
import pickle


df_train = pd.read_csv('cs-training.csv')
df_train.drop('Unnamed: 0', axis = 1, inplace = True)

df_train.columns = ['target', 'UnsecuredLines', 'age', 'PastDue30-59days', 'DebtRatio', 'MonthlyIncome', 'OpenCredit',
                   'Late_90days', 'RealEstate','PastDue60-89days', 'Dependents']

df_train['MonthlyIncome'].fillna(np.mean(df_train['MonthlyIncome']), inplace = True)
df_train['Dependents'].fillna(df_train['Dependents'].value_counts().argmax(), inplace = True)



from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

features = df_train.columns.tolist()
df_train.drop([features[2], features[4], features[5], features[6], features[8], features[10]], axis = 1, inplace = True)

X_train, X_valid, y_train, y_valid = train_test_split(np.array(df_train[df_train.columns[1:]]),
                                                      np.array(df_train['target']), random_state = 0)

clf = GradientBoostingClassifier(learning_rate = 1, max_depth = 3, n_estimators = 300, random_state = 0)
clf.fit(X_train, y_train)

pickle.dump(clf, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
probability = model.predict_proba(np.array([[0.766, 2, 0, 0]]))[:, 1]
print(probability)
