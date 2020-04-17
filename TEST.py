import _pickle as cPickle
import os
import sys


# for pack in os.listdir("src"):
#      sys.path.append(os.path.join("src", pack))
sys.path.append("./src/shared/")

import classes

with open('output/processed/test_data', 'rb') as f:
    set1 = cPickle.load(f)


with open('output/origin_origin/test_data', 'rb') as f:
    set2 = cPickle.load(f)

print(set1 == set2)
