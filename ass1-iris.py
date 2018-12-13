import sklearn

from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


# Create an object called iris with the iris data
iris = load_iris()

# Create a dataframe with the four feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add a new column with the species names, this is what we are going to try to predict
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

list_of_accu = []
for i in range(20):
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
    train, test = df[df['is_train']==True], df[df['is_train']==False]

    # Show the number of observations for the test and training dataframes
    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))

    # Create a list of the feature column's names
    features = df.columns[:4]
    y = pd.factorize(train['species'])[0]

    clf = RandomForestClassifier(n_jobs=1, random_state=0, min_samples_leaf = 5)
    clf.fit(train[features], y)
    y_pred = clf.predict(test[features])
    y_actual = pd.factorize(test['species'])[0]
    score = clf.score(test[features], y_actual)
    print("score for %d  is %.4f" %(i, score))
    list_of_accu.append(score)

    preds = iris.target_names[clf.predict(test[features])]
    print(test['species'].head())
    print(pd.crosstab(test['species'], preds, rownames=['Actual Species'], colnames=['Predicted Species']))

    #clf.predict_proba(test[features])




print("avg accuracy is : %.4f" %(np.mean(list_of_accu)))
