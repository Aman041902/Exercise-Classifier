import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
sys.path.append("../../src/models")
from LearningAlgo import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_pickle('../../data/interim/cluster_data.pkl')
df.info()

df_train = df.drop(['participant','category','set'],axis = 1)

x = df_train.drop('label',axis = 1)

y = df_train['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42,stratify=y)

basic_feat = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
sqa_feat = ['acc_r','gyro_r']
pca_feat =['pca_1','pca_2','pca_3']
time_feat = [f for f in df_train.columns if '_time_' in f]
freq_feat = [f for f in df_train.columns if ('_freq' in f) or ('_pse' in f)]
cluster_feat = ['cluster']

feat_set_1 = list(set(basic_feat))
feat_set_2 = list(set(basic_feat+sqa_feat+pca_feat))
feat_set_3 = list(set(feat_set_2+time_feat))
feat_set_4 = list(set(feat_set_3+ freq_feat+cluster_feat))

learner = ClassificationAlgorithms()
max_feat = 10

selected_features, ordered_features,ordered_scores = learner.forward_selection(max_feat,x_train,y_train)

selected_features = [
  "acc_z_freq_0.0_Hz_ws_14",
  "acc_x_freq_0.0_Hz_ws_14",
  "gyro_r_pse",
  "acc_y_freq_0.0_Hz_ws_14",
  "gyro_z_freq_0.714_Hz_ws_14",
  "gyro_r_freq_1.071_Hz_ws_14",
  "gyro_z_freq_0.357_Hz_ws_14",
  "gyro_x_freq_1.071_Hz_ws_14",
  "acc_x_max_freq",
  "gyro_z_max_freq"
]

feat_set = [
  feat_set_1,
  feat_set_2,
  feat_set_3,
  feat_set_4,
  selected_features
]

type(selected_features)

feat_names = ["feat_set_1", "feat_set_2", "feat_set_3", "feat_set_4", "selected features"]

iterations = 1
score_df = pd.DataFrame()


for i, f in zip(range(len(feat_set)), feat_names):
    print("Feature set:", i)
    selected_train_X = x_train[list(feat_set[i])]  # Convert set to list
    selected_test_X = x_test[list(feat_set[i])]    # Convert set to list

    # First run non deterministic classifiers to average their score.
    performance_test_nn = 0
    performance_test_rf = 0

    for it in range(0, iterations):
        print("\tTraining neural network,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.feedforward_neural_network(
            selected_train_X,
            y_train,
            selected_test_X,
            gridsearch=False,
        )
        performance_test_nn += accuracy_score(y_test, class_test_y)

        print("\tTraining random forest,", it)
        (
            class_train_y,
            class_test_y,
            class_train_prob_y,
            class_test_prob_y,
        ) = learner.random_forest(
            selected_train_X, y_train, selected_test_X, gridsearch=True
        )
        performance_test_rf += accuracy_score(y_test, class_test_y)

    performance_test_nn = performance_test_nn / iterations
    performance_test_rf = performance_test_rf / iterations

    # And we run our deterministic classifiers:
    print("\tTraining KNN")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.k_nearest_neighbor(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_knn = accuracy_score(y_test, class_test_y)

    print("\tTraining decision tree")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.decision_tree(
        selected_train_X, y_train, selected_test_X, gridsearch=True
    )
    performance_test_dt = accuracy_score(y_test, class_test_y)

    print("\tTraining naive bayes")
    (
        class_train_y,
        class_test_y,
        class_train_prob_y,
        class_test_prob_y,
    ) = learner.naive_bayes(selected_train_X, y_train, selected_test_X)

    performance_test_nb = accuracy_score(y_test, class_test_y)

    # Save results to dataframe
    models = ["NN", "RF", "KNN", "DT", "NB"]
    new_scores = pd.DataFrame(
        {
            "model": models,
            "feature_set": f,
            "accuracy": [
                performance_test_nn,
                performance_test_rf,
                performance_test_knn,
                performance_test_dt,
                performance_test_nb,
            ],
        }
    )
    score_df = pd.concat([score_df, new_scores])


score_df.sort_values("accuracy", ascending=False)

(
    class_train_y,
    class_test_y,
   class_train_prob_y,
   class_test_prob_y,
) = learner.random_forest(
        x_train[feat_set_4], y_train, x_test[feat_set_4], gridsearch=True
        )


accuracy = accuracy_score(y_test,class_test_y)


classes = class_test_prob_y.columns

matrix = confusion_matrix(y_test, class_test_y,labels=classes)

# create confusion matrix for cm
plt.figure(figsize=(10, 10))
plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = matrix.max() / 2.0
for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
    plt.text(
        j,
        i,
        format(matrix[i, j]),
        horizontalalignment="center",
        color="white" if matrix[i, j] > thresh else "black",
    )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.grid(False)
plt.show()

participant_df = df.drop(['set','category'],axis = 1)

x_train = participant_df[participant_df['participant']!='B'].drop(['label'],axis = 1)

y_train = participant_df[participant_df['participant']!='B']['label']

x_test = participant_df[participant_df['participant']=='B'].drop(['label'],axis = 1)

y_test = participant_df[participant_df['participant']=='B']['label']






