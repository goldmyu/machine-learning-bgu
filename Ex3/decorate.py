from sklearn.datasets import load_iris, load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np

# Create an object called iris with the iris data
# data = load_wine()

# Create a dataframe with the four feature variables
# df = pd.DataFrame(data.data, columns=data.feature_names)
# df['label'] = pd.Categorical.from_codes(data.target, data.target_names)
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data' )


c_size = 5
i_max = 10
r_size = 0.5


def compute_error(ensemble, x, y):
    pred = np.asarray([clf.predict_proba(x) for clf in ensemble])
    pred = np.average(pred, axis=0)
    pred = np.argmax(pred, axis=1)
    return 1 - (pred == y).mean()


def get_categorical_from_rnd(rnd, val_count):
    count = 0
    for x in range(len(val_count) - 1):
        if rnd < (count + val_count[x]):
            return x
        else:
            count = count + val_count[x]
    return len(val_count) -1


def generate_examples(x, creation_factor):
    examples_num = int(len(x) * creation_factor)
    categorical_columns = x.select_dtypes(include='object').columns
    categorical_data = pd.DataFrame(x, columns=categorical_columns)
    numerical_data = x.drop(columns = categorical_columns)
    means = numerical_data.mean(axis=0).values
    stds = numerical_data.std(axis=0).values
    numerical_examples = pd.DataFrame(columns=numerical_data.columns)
    for i,col in enumerate(numerical_data.columns):
        numerical_examples[col] = np.random.normal(means[i], stds[i] , examples_num)

    categorical_examples = generate_categorical_data( categorical_columns, categorical_data, examples_num)
    examples = pd.concat([categorical_examples, numerical_examples], axis=1)
    return examples


def generate_categorical_data(categorical_columns, categorical_data, examples_num):
    categorical_examples = pd.DataFrame(columns=categorical_columns)
    for i, col in enumerate(categorical_columns):
        column = categorical_data.iloc[:,i]
        column = pd.factorize(column, sort=True)[0]
        val_count = pd.Series(column).value_counts()
        randoms = np.random.randint(0, len(categorical_data[col]), examples_num)
        categorical_vec = []
        for x in range(len(randoms)):
            categorical_vec.append(get_categorical_from_rnd(randoms[x], val_count))
        categorical_examples[col] = categorical_vec
    return categorical_examples


def inverse_pred(row):
    new_row = []
    for x in row:
        new_row.append(x/ sum(row))
    return np.array(new_row)


def select_label(row):
    number = np.random.random()
    return get_categorical_from_rnd(number, row)


def label_examples(generated_x, ensemble):
    pred = np.asarray([clf.predict_proba(generated_x) for clf in ensemble])
    pred = np.average(pred, axis=0)
    pred[pred == 0] = 0.00001
    pred = 1/ pred
    predicts = np.apply_along_axis(inverse_pred, axis=1, arr=pred)
    labels = np.apply_along_axis(select_label, axis=1, arr=predicts)
    labels = pd.DataFrame(labels)
    return labels


def factorize_categorical_data(x_with_categorical):
    x = x_with_categorical.copy()
    categorical_columns = x.select_dtypes(include='object').columns
    for i, col in enumerate(categorical_columns):
        label_encoder = LabelEncoder()
        label_encoder.fit(x[col])
        x[col] = label_encoder.transform(x[col])
    return pd.DataFrame(x)


def run_decorate(dataset=df, c_size=c_size, i_max=i_max, creation_factor=r_size):
    i = 1
    trials = 1
    ensemble = []
    x_with_categorical = dataset.iloc[:, 0:-1]
    y = dataset.iloc[:, -1]
    y = pd.factorize(y)[0]
    x = factorize_categorical_data(x_with_categorical)
    clf = DecisionTreeClassifier(max_depth=2, min_samples_split=4)
    clf.fit(X=x, y=y)
    ensemble.append(clf)
    current_error = compute_error(ensemble, x, y)
    print("The initial error is : %.4f" % (current_error))
    while (i < c_size) & (trials < i_max):
        generated_x = generate_examples(x_with_categorical, creation_factor)
        generated_y = label_examples(generated_x, ensemble)
        x_full = pd.concat([x, generated_x])
        y_full = pd.concat([pd.DataFrame(y), generated_y])
        clf_iter = DecisionTreeClassifier(min_samples_split=4)
        clf_iter.fit(X=x_full, y=y_full)
        ensemble.append(clf_iter)
        error_iter = compute_error(ensemble, x, y)
        if error_iter < current_error:
            i += 1
            current_error = error_iter
            print("Found new ensemble, the error is : %.4f" % (current_error))
        else:
            ensemble.pop()
        trials += 1
    print("The final error is : %.4f" % (current_error))


run_decorate(df, c_size, i_max, r_size)