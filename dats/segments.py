from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras import backend as K
from dats.image_data_generator import DataGenerator
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.pyplot.switch_backend('Agg')
from PIL import Image


class localization:
    def __init__(self):
        self.id = list()
        self.defect = list()
        self.score = list()
        self.model_path = os.path.join(Path(os.getcwd()).parent.absolute(),
                                       'Severstal_API/models/unet.keras')
        self.jpg = os.path.join(Path(os.getcwd()).parent.absolute(),
                                'Severstal_API/static/bilder/localization.jpg')
        self.path = os.path.join('../../static', 'bilder/localization.jpg')
        self.file_loc = os.path.join(Path(os.getcwd()).parent.absolute(),
                                      'Severstal_API', 'stat_loc.csv')

    def dice_coef(self, y_true, y_pred, smooth=1):
        self.y_true = K.cast(y_true, 'float32')
        self.y_pred = K.cast(y_pred, 'float32')
        self.y_true_f = K.flatten(y_true)
        self.y_pred_f = K.flatten(y_pred)
        self.intersection = K.sum(self.y_true_f * self.y_pred_f)
        return (2. * self.intersection + smooth) / (
                K.sum(self.y_true_f) + K.sum(self.y_pred_f) + smooth)

    @property
    def res(self):
        model = load_model(self.model_path,
                           custom_objects={'dice_coef': self.dice_coef})
        test = pd.read_csv(os.path.join(Path(os.getcwd()).parent.absolute(),
                                        'Severstal_API/sample_submission.csv'))

        batches = DataGenerator(test)
        preds = model.predict(batches, steps=2, verbose=2)

        for i, batch in enumerate(batches):
            if i != 0: break
            plt.figure(figsize=(20, 36))
            for k in range(16):
                plt.subplot(16, 2, 2 * k + 1)
                img = batch[0][k,]
                img = Image.fromarray(img.astype('uint8'))
                img = np.array(img)
                dft = 0
                extra = '  has defect '
                msk_max = max([np.sum(preds[k, :, :, j]) for j in range(4)])
                for j in range(4):
                    msk = preds[k, :, :, j]
                    if np.sum(msk) == msk_max:
                        dft = j + 1
                        extra += ' ' + str(j + 1)
                if extra == '  has defect ': extra = ''
                plt.title(batches.df.iloc[k, 0], fontsize=18,
                          fontweight="bold")
                plt.axis('off')
                plt.imshow(img)
                plt.subplot(16, 2, 2 * k + 2)
                if dft != 0:
                    msk = preds[i + k, :, :, dft - 1]
                    plt.imshow(msk)
                else:
                    plt.imshow(np.zeros((128, 800)))
                plt.axis('off')
                mx = np.round(np.max(msk), 3)
                plt.title(batches.df.iloc[k, 0] + (extra +
                                                   '  (max pixel = ' + str(
                            mx) + ')'), fontsize=18, fontweight="bold")
                self.id.append(batches.df.iloc[k, 0])
                self.score.append(str(mx))
                self.defect.append(str(dft))
            stat_loc = {"id": self.id, "max_pixel": self.score, "defect":
                self.defect}
            pd.DataFrame(stat_loc).to_csv(self.file_loc,index=False)

            plt.subplots_adjust(wspace=0.2)

            plt.savefig(self.jpg, dpi=300)

        return self.path
