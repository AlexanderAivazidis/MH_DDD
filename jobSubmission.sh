#!/bin/sh
THEANO_FLAGS='device=cuda0,dnn.enabled=False,force_device=True,floatX=float32' python MultivariateGaussianMixtureTest.py
