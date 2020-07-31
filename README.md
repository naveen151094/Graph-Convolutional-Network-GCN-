# Graph-Convolutional-Network-GCN-
Graph Convolutional Network (GCN)

- The StellarGraph library supports many state-of-the-art machine learning algorithms on graphs 
- There are three sections:
    1. Data preparation using Pandas and scikit-learn: loading the graph from CSV les, doing
    some basic introspection, and splitting it into train, test and validation splits for ML
    2. Creating the GCN layers and data input using StellarGraph
    3. Training and evaluating the model using TensorFlow Keras, Pandas and scikit-learn.

Section-1:
- import the Python libraries.
- We retrieve a StellarGraph graph object holding this PubMedDiabetes dataset using the PubMedDiabetes loader from the datasets submodule.
- Splitting the dataset.

Section-2: Creating the GCN layers
- a data generator to convert the core graph structure and node features into a format that
  can be fed into the Keras model for training or prediction.
- A generator just encodes the information required to produce the model inputs. 

Section-3: Training and evaluating.
- Check its performance on the validation dataset.

How to run code : python gcn_model.py
