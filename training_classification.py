import rasterio
from rasterio import mask
from rasterio.features import rasterize
from rasterio.plot import reshape_as_image
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import mapping, Point, Polygon
from shapely.ops import cascaded_union

tile_path = r'/content/drive/MyDrive/Copy of T36UXV_20200406T083559_TCI_10m.jp2'

reader = rasterio.open(tile_path)
pic = reshape_as_image(reader.read())
fig = plt.figure(figsize=(15, 10))

train_df = gpd.read_file('/content/drive/MyDrive/masks/Masks_T36UXV_20190427.shp')
train_df = train_df.to_crs({'init': reader.meta['crs']['init']})

train_df = train_df.loc[~train_df["geometry"].isnull()]
train_df.head()
print(len(train_df))

failed = []
for num, row in train_df.iterrows():
    try:
        masked_image, out_transform = mask.mask(reader, [mapping(row['geometry'])], crop=True, nodata=0)
    except Exception as e:
        # print(e)
        failed.append(num)
print("Rasterio failed to mask {} files".format(len(failed)))
print(failed)


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

mask = rasterize(shapes=poly_shp, out_shape=im_size)


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


def get_tf_dataset(dataset_, indexing, augment=False, include_mask=False):
    def data_generator_():
        for i in indexing:
            window = dataset_[0][i]
            x_slice = slice(window[0, 0], window[1, 0])
            y_slice = slice(window[0, 1], window[1, 1])
            mask_piece = mask[x_slice, y_slice]
            yield (pic[x_slice, y_slice], mask_piece, dataset_[1][i])

    out = tf.data.Dataset.from_generator(data_generator_,
                                         output_signature=(tf.TensorSpec(shape=INPUT_SIZE + [3], dtype=tf.uint8),
                                                           tf.TensorSpec(shape=INPUT_SIZE, dtype=tf.uint8),
                                                           tf.TensorSpec(shape=(), dtype=tf.int32)))
    if augment:
        transformation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GaussNoise(p=0.8, var_limit=(5, 10))
        ])

        def transform(image, mask, target):
            res = transformation(image=image, mask=mask)
            return res["image"], res["mask"], target

        out = out.map(lambda x, y, z: tf.numpy_function(transform,
                                                        inp=[x, y, z],
                                                        Tout=(tf.uint8, tf.uint8, tf.int32)))
    out = out.map(lambda x, y, z: (tf.cast(x, tf.float16) / 255.0,
                                   tf.cast(y, tf.float16),
                                   tf.cast(z, tf.float16)
                                   ))
    if not include_mask:
        out = out.map(lambda x, y, z: (tf.reshape(x, INPUT_SIZE + [3]), z))
    else:
        out = out.map(lambda x, y, z: (tf.reshape(x, INPUT_SIZE + [3]), tf.reshape(y, INPUT_SIZE), z))
    out = out.batch(BATCH_SIZE).prefetch(AUTO)
    return out


def get_model(weights=None):
    inp = tf.keras.Input(INPUT_SIZE + [3])
    model_base = sm.Unet(BACKBONE, classes=1, activation='linear')
    inter = model_base(inp)

    inter = tf.keras.layers.Activation('swish')(inter)
    inter = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(inter)
    inter = tf.keras.layers.Flatten()(inter)
    targeting = tf.keras.layers.Dense(1, activation='sigmoid', name='tar')(inter)

    model = tf.keras.models.Model(inputs=inp, outputs=targeting)

    model.compile('Adam', loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    if weights is not None:
        model.load_weights(weights)
    return model


model = get_model('/content/drive/MyDrive/weights_0.h5')

def lr_schedule(epoch):
    lrs = [1E-3, 5E-4, 3E-4, 1E-4, 5E-5, 3E-5, 1E-5]
    if epoch in range(len(lrs)):
        return lrs[epoch]
    else:
        return 3E-6


lr_sch = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=False)
m_chkp = tf.keras.callbacks.ModelCheckpoint(f"./weights_0_class.h5",
                                            monitor='val_accuracy',
                                            verbose=0,
                                            save_best_only=True,
                                            save_weights_only=True,
                                            mode='max',
                                            save_freq='epoch')

train_idx, val_idx = train_test_split(np.arange(DATA_LEN), test_size=0.2, shuffle=True, random_state=31)
data_train = get_tf_dataset(dataset_raw, train_idx,
                            augment=True, include_mask=True)
data_val = get_tf_dataset(dataset_raw, val_idx,
                          augment=False, include_mask=True)

model.fit(data_train, epochs=10, verbose=1, validation_data=data_val,
          callbacks=[lr_sch, m_chkp])

