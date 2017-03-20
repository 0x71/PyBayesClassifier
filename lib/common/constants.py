# Copyright (C) 2016-2017  Nils Rogmann.
# This file is part of PyBayesClassifier.
# See the file 'docs/LICENSE' for copying permission.

from os import getcwd

APP_ROOT = getcwd()

TRAIN_RATIO = 0.6
TEST_RATIO = 0.2

THRESHOLD = 0.001

CLASS_THRESHOLD = 0.10e-70
CLASS_THRESHOLD_WEIGHT = {"dns": 20.0, "icmp": 8.0}
CLASS_THRESHOLD_WEIGHT_OLD = 10.0

# 10.0, 3.0
# 15.0, 3.0
# 20.0, 5.0


# Best 20.0, 5.0
