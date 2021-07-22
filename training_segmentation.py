
import rasterio
from rasterio import mask
from rasterio.features import rasterize
from rasterio.plot import reshape_as_image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import geopandas as gpd
from shapely.geometry import mapping, Point, Polygon
from shapely.ops import cascaded_union

# Data Preparation

tile_path = r'/content/drive/MyDrive/Work/QuantumTasks/Quantum internship/T36UXV_20200406T083559_TCI_10m.jp2'

reader = rasterio.open(tile_path)
PIC = reshape_as_image(reader.read())
fig = plt.figure(figsize=(15, 10))


train_df = gpd.read_file('/content/drive/MyDrive/Work/QuantumTasks/Quantum internship/masks/Masks_T36UXV_20190427.shp')
train_df = train_df.to_crs({'init': reader.meta['crs']['init']})

train_df = train_df.loc[~train_df["geometry"].isnull()]


def poly_from_utm(polygon, transform):
    poly_pts = []

    # make a polygon from multipolygon
    poly = cascaded_union(polygon)
    for i in np.array(poly.exterior.coords):
        # transfrom polygon to image crs, using raster meta
        poly_pts.append(~transform * tuple(i))

    # make a shapely Polygon object
    new_poly = Polygon(poly_pts)
    return new_poly


poly_shp = []
im_size = (reader.meta['height'], reader.meta['width'])
for num, row in train_df.iterrows():
    if row['geometry'].geom_type == 'Polygon':
        poly = poly_from_utm(row['geometry'], reader.meta['transform'])
        poly_shp.append(poly)
    else:
        for p in row['geometry']:
            poly = poly_from_utm(p, src.meta['transform'])
            poly_shp.append(poly)

MASK = rasterize(shapes=poly_shp, out_shape=im_size)


def get_raw_dataset(data, size=(320, 320), margin=(100, 100)):
    img, mask = data
    windows = []
    targets = []
    window = np.array([[0, 0], [size[0], size[1]]])
    shift_right = np.array([[margin[0], 0], [margin[0], 0]])
    new_row = lambda: window * np.array([[0, 1], [0, 1]]) + np.array([[0, margin[1]], [size[0], margin[1]]])
    cut_out = lambda: (img[window[0, 0]:window[1, 0], window[0, 1]:window[1, 1], :],
                       mask[window[0, 0]:window[1, 0], window[0, 1]:window[1, 1]])
    while True:

        if window[1, 0] > img.shape[0]:
            window = new_row()
            continue
        if window[1, 1] > img.shape[1]:
            break

        windows.append(window.copy())
        tar = np.max(mask[window[0, 0]:window[1, 0], window[0, 1]:window[1, 1]])

        targets.append(tar)
        window += shift_right
    return (np.array(windows), np.array(targets))


dataset_raw = get_raw_dataset((pic, mask))
dataset_raw = (dataset_raw[0][dataset_raw[1] == 1], dataset_raw[1][dataset_raw[1] == 1])



import segmentation_models as sm

sm.set_framework('tf.keras')
sm.framework()
import tensorflow as tf
from tensorflow import keras
import albumentations as A
import efficientnet as efn
import sys

from sklearn.model_selection import train_test_split


DATA_LEN = len(dataset_raw[0])

BACKBONE = 'efficientnetb3'
INPUT_SIZE = [320, 320]
BATCH_SIZE = 8
AUTO = tf.data.experimental.AUTOTUNE


def get_tf_dataset_seg(dataset_, indexing, augment=False):
    def data_generator_():
        for i in indexing:
            yield dataset_[0][i]

    out = tf.data.Dataset.from_generator(data_generator_,
                                         output_signature=tf.TensorSpec(shape=[2, 2], dtype=tf.int32))

    def main_window_map(window):
        x_slice = slice(window[0, 0], window[1, 0])
        y_slice = slice(window[0, 1], window[1, 1])
        return PIC[x_slice, y_slice], MASK[x_slice, y_slice]

    out = out.map(lambda x: tf.numpy_function(main_window_map,
                                              inp=[x], Tout=(tf.uint8, tf.uint8)),
                  num_parallel_calls=AUTO)

    if augment:
        def transform(image, mask):
            seed = tf.random.uniform((), maxval=2048)
            res_im = tf.image.random_flip_left_right(image, seed=seed)
            res_im = tf.image.random_flip_up_down(res_im, seed=seed + 1)
            res_ms = tf.image.random_flip_left_right(mask, seed=seed)
            res_ms = tf.image.random_flip_up_down(res_ms, seed=seed + 1)
            return res_im, res_ms

        out = out.map(lambda x, y: transform(x, y))
    out = out.map(lambda x, y: (tf.cast(x, tf.float16) / 255.0,
                                tf.cast(y, tf.float16)
                                ))
    out = out.map(lambda x, y: (tf.reshape(x, INPUT_SIZE + [3]),
                                tf.reshape(y, INPUT_SIZE)))
    out = out.repeat().batch(BATCH_SIZE).prefetch(AUTO)
    return out

#########
# Modelling
#########

def get_model_seg():
    model = sm.Unet(BACKBONE, classes=1, activation='sigmoid')
    model.compile('Adam', loss=sm.losses.bce_jaccard_loss,
                  metrics=[sm.metrics.iou_score])
    return model

with strategy.scope():
    model = get_model_class()


def lr_schedule(epoch):
    lrs = [1E-3, 5E-4, 3E-4, 1E-4, 5E-5, 3E-5, 1E-5]
    if epoch in range(len(lrs)):
        return lrs[epoch]
    else:
        return 3E-6


lr_sch = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=False)
m_chkp = tf.keras.callbacks.ModelCheckpoint(f"./weights_0_seg.h5",
                                            monitor='val_iou_score',
                                            verbose=0,
                                            save_best_only=True,
                                            save_weights_only=True,
                                            mode='max',
                                            save_freq='epoch')

train_idx, val_idx = train_test_split(np.arange(DATA_LEN), test_size=0.2, shuffle=True, random_state=31)
train_steps = 2 * len(train_idx) // BATCH_SIZE + 1
val_steps = 2 * len(val_idx) // BATCH_SIZE + 1

data_train = get_tf_dataset_seg(dataset_raw, train_idx,
                                augment=True)
data_val = get_tf_dataset_seg(dataset_raw, val_idx,
                              augment=False)

#####
# Training
#####

model.fit(data_train, steps_per_epoch=train_steps,
          epochs=10, verbose=1,
          validation_data=data_val, validation_steps=val_steps,
          callbacks=[lr_sch, m_chkp])

