import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import recall_score, precision_score
from sklearn.utils import resample


data = pd.read_csv('images.csv')
X = data.drop(columns='label')
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_y_train = X_train.copy(deep=True)
X_y_train['y'] = y

# group_n = 0
# fig, axs = plt.subplots(10, 10, figsize=(10, 10),
#                         sharex=True, sharey=True)
# for _, group in X_y_train.groupby('y', as_index=False):
#     images = [x.values for _, x in group.sample(10).drop(columns='y').iterrows()]
#     image_n = 0
#     for image in images:
#         ax = axs[group_n][image_n]
#         ax.axis('off')
#         ax.imshow(image.reshape(28, 28).astype('uint8'))
#         image_n += 1
#     group_n += 1
# plt.show()

# sns.heatmap(pd.DataFrame(X_train.std().values.reshape(28, 28)))
# plt.show()

selector = VarianceThreshold(threshold=0)
selector.fit(X_train)
useful_factors = selector.get_feature_names_out()
X_train = X_train[useful_factors]
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0)

model_decision_tree = DecisionTreeClassifier(min_samples_leaf=10,
                                             criterion='gini',
                                             random_state=0)
model_decision_tree.fit(X_train, y_train)
print(model_decision_tree.score(X_val, y_val))

model_random_forest = RandomForestClassifier(n_estimators=100,
                                             min_samples_leaf=3,
                                             max_features=int(len(list(X_train.columns)) ** (1 / 2)),
                                             criterion='gini',
                                             random_state=0)
model_random_forest.fit(X_train, y_train)
print(model_random_forest.score(X_val, y_val))
# y_val_pred = model_random_forest.predict(X_val)
# print(recall_score(y_val, y_val_pred, average='macro'))
# print(precision_score(y_val, y_val_pred, average='macro'))

model_gradient_boosting = GradientBoostingClassifier(n_estimators=100,
                                                     min_samples_leaf=3,
                                                     max_features=int(len(list(X_train.columns)) ** (1 / 2)),
                                                     random_state=0)
model_gradient_boosting.fit(X_train, y_train)
print(model_gradient_boosting.score(X_val, y_val))

X_test = X_test[useful_factors]
accuracy_scores = list()
for i in range(100):
    X_cur, y_cur = resample(X_test, y_test, random_state=i)
    accuracy_scores.append(model_random_forest.score(X_cur, y_cur))
accuracy_scores.sort()
accuracy_scores = accuracy_scores[3:97]
print(accuracy_scores[0], accuracy_scores[-1])
print(model_random_forest.score(X_test, y_test))
