AnnexML
=======

Approximate Nearest Neighbor Search for Extreme Multi-Label Classification


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


Copyright &copy; 2017 Yahoo Japan Corporation All Rights Reserved.
