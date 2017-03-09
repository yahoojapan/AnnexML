AnnexML: Approximate Nearest Neighbor Search for Extreme Multi-Label Classification
===================================================================================

AnnexML is a multi-label classifier designed for extremely large label space (10^4 to 10^6).
At training step, AnnexML constructs k-nearest neighbor graph of the label vectors and attempts to reproduce the graph structure in the embedding space.
The prediction is efficiently performed by using an approximate nearest neighbor search method which efficiently explores the learned k-nearest neighbor graph in the embedding space.


Build
-----

A recent compiler supporting C++11 is required.

    $ make -C src/ annexml


Usage
-----

### Data Format

AnnexML takes multi-label svmlight / libsvm format.
The datasets on [The Extreme Classification Repository](https://manikvarma.github.io/downloads/XC/XMLRepository.html), which have an additional header line, are also applicable.

    32,50,87 1:1.9 23:0.48 79:0.63
    50,51,126 4:0.71 23:0.99 1005:0.08 1018:2.15


### Training and prediction

Model parameters and some file paths are specified in a JSON file or command line arguments.
The settings specified in arguments will overwrite those in the JSON file.
Recommended parameters are in `annexml-example.json`.

Examples of training:

    $ src/annexml train annexml-example.json
    $ src/annexml train annexml-example.json num_thread=32   # use 32 CPU threads for training
    $ src/annexml train annexml-example.json cls_type=0   # use k-means clustering for data partitioning

Examples of prediction:

    $ src/annexml predict annexml-example.json
    $ src/annexml predict annexml-example.json num_learner=4 num_thread=1   # use only 4 learners and 1 CPU thread for prediction
    $ src/annexml predict annexml-example.json pred_type=0   # use brute-force cosine calculation

Usage of the evaluation script written in python is as follow:

    $ cat annexml-result-example.txt | python scripts/learning-evaluate_predictions.py
    6616
    [5724, 4905, 4116, 3491, 2998]
    [5724.0, 5407.167550874974, 5104.534229455956, 4833.8975471349, 4593.97654093441]
    P@1=0.865175
    P@2=0.803280
    P@3=0.742896
    P@4=0.689087
    P@5=0.641898
    nDCG@1=0.865175
    nDCG@2=0.817287
    nDCG@3=0.771544
    nDCG@4=0.730637
    nDCG@5=0.694374


#### Model Parameters and File Paths

    emb_size          Dimension size of embedding vectors
    num_learner       Number of learners (or models) for emsemble learning
    num_nn            Number of (approximate) nearest neighbors used in training and prediction
    cls_type          Algorithm type used for data partitioning
                      1 : learning procedure which finds min-cut of approximate KNNG
                      0 : k-means clustering
    cls_iter          Number of epochs for data partitioning algorithms
    emb_iter          Number of epochs for learning embeddings
    label_normalize   Label vectors are normalized or not
    eta0              Initial value of AdaGrad learning rate adjustement
    lambda            L1-regularization parameter of data partitioning (only used if cls_type = 1)
    gamma             Scaling parameter for cosine ([-1, 1] to [-gamma, gamma]) in learning embeddings
    pred_type         Algorithm type used for prediction of k-nearest neighbor classifier
                      1 : approximate nearest neighbor search method which explores learned KNNG
                      0 : brute-force calculation
    num_edge          Number of direct edges per vertex in learned KNNG (only used if pred_type = 1)
    search_eps        Parameter for exploration of KNNG (only used if pred_type = 1)
    num_thread        Number of CPU threads used in training and prediction
    seed              Random seed
    verbose           Vervosity level (ignore if num_thread > 1)

    train_file        File path of training data
    predict_file      File path of prediction data
    model_file        File path of output model
    result_file       File path of prediction result


License
-------

Copyright (C) 2017 Yahoo Japan Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Dependencies
------------

AnnexML includes the following software.

- (2-clause BSD license) [picojson](https://github.com/kazuho/picojson)


Copyright &copy; 2017 Yahoo Japan Corporation All Rights Reserved.
