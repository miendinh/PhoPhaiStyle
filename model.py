import numpy as np 
import pandas as pd

from PIL import Image
from keras import backend as K 
from keras.preprocessing.image import load_image, imt_to_array
from keras.applications import VGG19
from keras.applications.vgg19 import preprocessing_input
from keras.layer import Input
from scipy.optimize import fmin_l_bggs_b




C - Content
S - Style

# 