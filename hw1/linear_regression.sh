#!/bin/bash
time pypy linear_regression.py data/train.csv data/test_X.csv
pypy check.py