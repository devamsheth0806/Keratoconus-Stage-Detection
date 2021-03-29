import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.preprocessing import LabelEncoder

labels = pd.read_csv('labels.csv')
dataset = pd.read_csv('dataset.csv')
dataset.drop(['Unnamed: 0'], axis=1, inplace=True)
dataset.drop(['En.Anterior.'], axis=1, inplace=True)
dataset.drop(['idEye'], axis=1, inplace=True)
labels.drop(['Unnamed: 0'], axis=1, inplace=True)
labels.drop(['Data.PLOS_One.idEye'], axis=1, inplace=True)
# for i in dataset.iloc[2]:
#     print(type(i))
# print(dataset.columns)
# train = pd.read_csv('labels.csv')
# test = pd.read_csv('dataset.csv')
dataset.apply(LabelEncoder().fit_transform)
labels.apply(LabelEncoder().fit_transform)

my_feature_columns = []
for key in dataset.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))


def input_fn(features, labels, training=True, batch_size=500):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)


classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns,
                                        hidden_units=[1024, 512, 256],
                                        n_classes=5)

print(dataset.head())
# Train the Model.
classifier.train(input_fn=lambda: input_fn(dataset, labels['clster_labels'], training=True), steps=4901)

eval_result = classifier.evaluate(input_fn=lambda: input_fn(dataset, labels['clster_labels'], training=False))
acc = eval_result['accuracy']*100
print('\nTest set accuracy: {0:0.2f}%\n'.format(acc))

# save
# saver = tf.train.Saver({"classifier":classifier})
# saver.save(sess, 'my-model')
# new_saver = tf.train.import_meta_graph('my-model.meta')
# new_saver.restore(sess, 'my-model')