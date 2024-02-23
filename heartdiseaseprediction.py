import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split, trian_test_split
from scipy.stats import randint

df = pd.read_csv('/heart.csv')

print(df.head())

print(df.describe())

X = df.drop('target', axis = 1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

param_dist = {
    'max_depth': randint(1,20),
    'min_samples_split': randint(2,20),
    'min_samples_leaf': randint(1,20)
}

random_search = RandomizedSearchCV(DecisionTreeClassifier(random_state = 42),
                                   param_distributions= param_dist,
                                   n_iter = 100,
                                   cv = 5,
                                   random_state = 42)

random_search.fit(X_train, y_train)
best_tree = random_search.best_estimator_

cross_val_scores_random = cross_val_score(best_tree, X,y, cv = 5)

print(f"Cross validation scores: {cross_val_scores_random}")
print(f"Mean cross validation score: {cross_val_scores_random.mean()}")