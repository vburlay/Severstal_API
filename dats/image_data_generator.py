import keras._tf_keras.keras
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
from segmentation_models import  get_preprocessing

class DataGenerator(keras.utils.Sequence):
    def __init__(self, df, batch_size = 16, subset="prod", shuffle=False,
                 info={}):
        super().__init__()
        self.df = df
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.preprocess = get_preprocessing('resnet50')
        self.info = info
        self.data_path = os.path.join(
            Path(Path(os.getcwd()).parent.absolute(), 'Severstal_API/data'))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def rle2maskResize(rle):
        # CONVERT RLE TO MASK
        if (pd.isnull(rle)) | (rle == ''):
            return np.zeros((128, 800), dtype=np.uint8)

        height = 256
        width = 1600
        mask = np.zeros(width * height, dtype=np.uint8)

        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2] - 1
        lengths = array[1::2]
        for index, start in enumerate(starts):
            mask[int(start):int(start + lengths[index])] = 1

        return mask.reshape((height, width), order='F')[::2, ::2]

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        X = np.empty((self.batch_size,128,800,3),dtype=np.float32)
        y = np.empty((self.batch_size,128,800,4),dtype=np.int8)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        for i,f in enumerate(self.df['ImageId'].iloc[indexes]):
            self.info[index*self.batch_size+i]=f
            X[i,] = Image.open(os.path.join(self.data_path, f)).resize((800,
                                                                        128))

        if self.preprocess!=None: X = self.preprocess(X)
        if self.subset == 'prod': return X, y
        else: return (X,)