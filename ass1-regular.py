from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data' )



list_of_accu = []
for i in range(20):
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
    train, test = df[df['is_train']==True], df[df['is_train']==False]

    df_train_data = train.iloc[:, 0:-2]
    df_train_labels = train.iloc[:, -2]
    df_test_data = test.iloc[:, 0:-2]
    df_test_labels = test.iloc[: ,-2]

    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))

    # We change those parameters for each dataset
    max_features = "auto"
    min_samples_leaf=5
    max_depth=20
    n_estimators=10
    clf = RandomForestClassifier(n_jobs=1, random_state=0, min_samples_leaf =min_samples_leaf, max_features=max_features,max_depth=max_depth,n_estimators=n_estimators )
    clf.fit(df_train_data, df_train_labels)
    y_pred = clf.predict(df_test_data)
    score = clf.score(df_test_data, df_test_labels)
    print("score for %d  is %.4f" %(i, score))
    list_of_accu.append(score)

    preds = clf.predict(df_test_data)
    print(pd.crosstab(df_test_labels, preds, rownames=['Actual labels'], colnames=['Predicted labels']))


print("avg accuracy is : %.4f" %(np.mean(list_of_accu)))
