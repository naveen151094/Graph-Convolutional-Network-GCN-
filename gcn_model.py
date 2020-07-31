#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 13:53:15 2020

@author: naveen
"""

'''
- The StellarGraph library supports many state-of-the-art machine learning algorithms on graphs 
- There are three sections:
    1. Data preparation using Pandas and scikit-learn: loading the graph from CSV les, doing
    some basic introspection, and splitting it into train, test and validation splits for ML
    2. Creating the GCN layers and data input using StellarGraph
    3. Training and evaluating the model using TensorFlow Keras, Pandas and scikit-learn'''

'''
Section-1:
- import the Python libraries.
'''
import stellargraph as sg
import pandas as pd
import os
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, model_selection
from IPython.display import display, HTML
import matplotlib.pyplot as plt

'''
We retrieve a StellarGraph graph object holding this PubMedDiabetes dataset using the PubMedDiabetes loader
from the datasets submodule.
'''
dataset = sg.datasets.PubMedDiabetes()  # load PubMedDiabetes dataset
#dataset = sg.datasets.CiteSeer()
display(HTML(dataset.description))
G, node_subjects = dataset.load()

''''
The info method can help us verify that our loaded graph matches the description:
'''
print(G.info())

# Counts of each label in dataframe.
print(node_subjects.value_counts().to_frame())

'''
Splitting the dataset:
- Here we're taking 2000 node labels for training, 600 for validation, and the rest for testing using stratied sampling.
'''
train_subjects, test_subjects = model_selection.train_test_split(node_subjects,train_size=250, test_size=None, stratify=node_subjects)
val_subjects, test_subjects = model_selection.train_test_split(test_subjects,train_size=750, test_size=None, stratify=test_subjects)
train_subjects.value_counts().to_frame()

'''
Converting to numeric arrays for categorical data - we can use the LabelBinarizer transform or one hot encoder or dummies in pandas 
'''
target_encoding = preprocessing.LabelBinarizer()
train_targets = target_encoding.fit_transform(train_subjects)
val_targets = target_encoding.transform(val_subjects)
test_targets = target_encoding.transform(test_subjects)

'''
Section-2: Creating the GCN layers
- a data generator to convert the core graph structure and node features into a format that
  can be fed into the Keras model for training or prediction.
- A generator just encodes the information required to produce the model inputs. 
'''
generator = FullBatchNodeGenerator(G, method="gcn")
train_gen = generator.flow(train_subjects.index, train_targets)

'''
Build the GCN layers:
- with 2 hidden layer and with 10 units in each and both layers uses relu activation with 30% dropout.
'''
gcn = GCN(layer_sizes=[10,10], activations=["relu", "relu"], generator=generator, dropout=0.1)

'''
we now expose the input and output tensors of the GCN model for node prediction.
'''
x_inp, x_out = gcn.in_out_tensors()
print(x_out)

'''
The predicted class is the element with the highest probability value.
'''
predictions = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

'''
Section-3: Training and evaluating
'''

model = Model(inputs=x_inp, outputs=predictions)
model.compile(
    optimizer=optimizers.Adam(lr=0.01),
    loss=losses.categorical_crossentropy,
    metrics=["acc"],
)
'''
Check its performance on the validation dataset.
'''
val_gen = generator.flow(val_subjects.index, val_targets)

'''
EarlyStopping, stop training if the validation accuracy stops improving.
'''
from tensorflow.keras.callbacks import EarlyStopping
es_callback = EarlyStopping(monitor="val_acc", patience=60, restore_best_weights=True)

'''
train the model with training data and validation data.
'''
history = model.fit(
    train_gen,
    epochs=200,
    validation_data=val_gen,
    verbose=2,
    shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
    callbacks=[es_callback])

'''
we can view the behaviour loss function and any other metrics using the plot_history function.
'''
sg.utils.plot_history(history)

'''
Validate the model on test datasrt and evaluate with accuracy and loss.
'''
test_gen = generator.flow(test_subjects.index, test_targets)
test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))
    
'''
try predictions on all nodes.
'''
all_nodes = node_subjects.index
all_gen = generator.flow(all_nodes)
all_predictions = model.predict(all_gen)
node_predictions = target_encoding.inverse_transform(all_predictions.squeeze())

df = pd.DataFrame({"Predicted": node_predictions, "True": node_subjects})
print(df.head(20))
df.to_csv('predicted_true_label.csv')

'''
embeddings captures the information about the nodes and their neighbourhoods in the form a neural network that produces those vectors.  
'''
embedding_model = Model(inputs=x_inp, outputs=x_out)
emb = embedding_model.predict(all_gen)
print(emb.shape)

'''
16 dimensional plot, which is hard for humans to visualise,
Hence we reduced 16-dimensions to 2-dimensions using PCA or TSNE methods.
'''
from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
transform = PCA  # or TSNE
X = emb.squeeze(0)
print(X.shape)
trans = transform(n_components=2)
X_reduced = trans.fit_transform(X)
print(X_reduced.shape)

'''
scatter plot shows good clustering, where nodes of a single colour are mostly grouped together.
'''
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    c=node_subjects.astype("category").cat.codes,
    cmap="jet",
    alpha=0.7,
)
ax.set(
    aspect="equal",
    xlabel="$X_1$",
    ylabel="$X_2$",
    title=f"{transform.__name__} visualization of GCN embeddings for PubMedDiabetes dataset",
)