# Exemplar Models

This repository contains an implementation of exemplar models for the purpose of approximate density estimation.


The model is detailed in the following paper:

[EX2: Exploration with Exemplar Models for Deep Reinforcement Learning](https://arxiv.org/abs/1703.01260)

Justin Fu*, John D. Co-Reyes*, Sergey Levine. NIPS 2017

Setup
----

This library requires Numpy, Scipy, Tensorflow, and Matplotlib.


Running the Code
----

To run the demo code, simply run the following script:
```
python scripts/twod_example.py
```

This will save a series of images taken over the course of training into the `data` folder. The model's estimates are shown in the left panel, and the ground truth data is shown on the right.

Iteration 1000:

![1000](https://github.com/justinjfu/exemplar_models/blob/master/imgs/twod_itr1000.png)

Iteration 10000:

![10000](https://github.com/justinjfu/exemplar_models/blob/master/imgs/twod_itr10000.png)

Iteration 20000:

![20000](https://github.com/justinjfu/exemplar_models/blob/master/imgs/twod_itr20000.png)

Iteration 30000:

![30000](https://github.com/justinjfu/exemplar_models/blob/master/imgs/twod_itr30000.png)
