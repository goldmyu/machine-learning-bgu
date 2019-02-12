from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

df = pd.read_csv('C:\ex3\wineQualityReds_no_index.csv' )


c_size = 5
i_max = 10
r_size = 0.5


def compute_error(ensemble, x, y):
    pred = predict_ensemble(ensemble, x)
    return 1 - (pred == y).mean()


def predict_ensemble(ensemble, x):
    pred = predict_proba_ensemble(ensemble, x)
    pred = np.argmax(pred, axis=1)
    return pred


def predict_proba_ensemble(ensemble, x):
    pred = np.asarray([clf.predict_proba(x) for clf in ensemble])
    pred = np.average(pred, axis=0)
    return pred


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
    categorical_columns = x.select_dtypes(include='int').columns
    numerical_data = x.drop(columns = categorical_columns)
    means = numerical_data.mean(axis=0).values
    stds = numerical_data.std(axis=0).values
    numerical_examples = pd.DataFrame(columns=numerical_data.columns)
    for i,col in enumerate(numerical_data.columns):
        numerical_examples[col] = np.random.normal(means[i], stds[i] , examples_num)

    categorical_examples = generate_categorical_data( categorical_columns, x, examples_num)
    examples = pd.concat([categorical_examples, numerical_examples], axis=1)
    return examples


def generate_categorical_data(categorical_columns, x, examples_num):
    categorical_data = pd.DataFrame(x, columns=categorical_columns)
    categorical_examples = pd.DataFrame(columns=categorical_columns)
    for i, col in enumerate(categorical_columns):
        val_count = categorical_data[col].value_counts()
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


def run_decorate(x, y, c_size, i_max, creation_factor):
    i = 1
    trials = 1
    ensemble = []
    clf = DecisionTreeClassifier(max_depth=2, min_samples_split=4)
    clf.fit(X=x, y=y)
    ensemble.append(clf)
    current_error = compute_error(ensemble, x, y)
    print("The initial error is : %.4f" % (current_error))
    while (i < c_size) & (trials < i_max):
        generated_x = generate_examples(x, creation_factor)
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
    return ensemble


def run_10_fold_decorate(dataset=df, c_size=c_size, i_max=i_max, creation_factor= r_size):
    kf= KFold(n_splits=10,shuffle=True)
    x_with_categorical = dataset.iloc[:, 0:-1]
    x = factorize_categorical_data(x_with_categorical)
    y = dataset.iloc[:, -1]
    y = pd.factorize(y)[0]
    precisions = []
    accuracies = []
    recalls = []
    recalls_macro=[]
    for train_index, test_index in kf.split(x):
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        ensemble = run_decorate(X_train, y_train, c_size, i_max, creation_factor)
        predicts = predict_ensemble(ensemble, X_test)
        precisions.append(precision_score(y_test, predicts, average='micro'))
        accuracies.append(accuracy_score(y_test,predicts))
        recalls.append(recall_score(y_test,predicts, average='micro'))
        recalls_macro.append(recall_score(y_test, predicts, average='macro'))
    print("precisions")
    print(*precisions)
    print("accuracies")
    print(*accuracies)
    print("recalls")
    print(*recalls)
    print("recalls_macro")
    print(*recalls_macro)


run_10_fold_decorate(df, c_size, i_max, r_size)