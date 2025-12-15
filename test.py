import os
from os import listdir
from uniface import RetinaFace, Landmark106, ArcFace
from uniface.face_utils import face_alignment
from uniface.constants import RetinaFaceWeights, ArcFaceWeights
import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm



df = pd.read_parquet("faceEmbeddings.parquet")
