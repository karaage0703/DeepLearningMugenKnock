# Kerasを使ったった

#### コードをなるべくスクラッチで書く

- [インポート](#import)
- [モデル定義](#model)
- [モデル定義(KerasのAPI)](#api-model)
- [最適化の設定](#optimize)
- [学習データセットの用意](#dataset)
- [学習](#train)
- [学習モデルの保存](#save)
- [テスト](#test)
- [コード](#code)

#### [KerasのAPIを使う](#api)

#### [エラー集](#error)

## <a id="import">1. インポート</a>


最初は必要なものをpipでインストールする。

```bash
$ pip install tensorflow keras argparse opencv-python numpy glob
```

GPUでtensorlfowを使う場合は、tensorflow_gpuをインストールする必要がある。

コードではtensorflowをimport する。tfとエイリアスをつけることが多い。

```python
import tensorflow as tf
import keras
```

あとは必要なものももろもろimportする。

```python
import argparse
import cv2
import numpy as np
from glob import glob
```

GPU使用のときはこの記述をする。
*config.gpu_options.allow_growth = True* はGPUのメモリを必要な分だけ確保しますよーという宣言。
*config.gpu_options.visible_device_list="0"* は１個目のGPUを使いますよーっていう宣言。

```python
# GPU config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list="0"
sess = tf.Session(config=config)
K.set_session(sess)
```

次に諸々必要な宣言をする。num_classesは分類するクラス数。今回はアカハライモリ(akahara)とマダライモリ(madara)の２クラス。img_height, img_widthは入力する画像のサイズ。

```python
num_classes = 2
img_height, img_width = 64, 64
channel = 3
```


## <a id="model">2. モデル定義</a>

Kerasはこのように書ける。ほとんどkerasで用意されているのでらくらｋ。

```python
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization

def Mynet():
    inputs = Input((img_height, img_width, channel))
    x = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1_1')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1_2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2_1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2_2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3_1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3_2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4_1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4_2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(1024, name='dense1', activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, name='dense2', activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x, name='model')
    return model
```

## <a id="api-model">モデル定義(Keras API)</a>

KerasのAPIによるモデルを使う方法。例えばResNet50を使う方法。

```python
def model():
    from keras.applications.resnet50 import ResNet50
    model = ResNet50(include_top=False,
        weights='imagenet',
        input_shape=(img_height, img_width, channel))

    last = model.output
    x = Flatten()(last)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(cf.Class_num, activation='softmax')(x)
    model = Model(model.input, x)
    return model
```

## <a id="optimize">3. Optimizerの設定</a>

モデルを書いたら次に最適化optimizerを設定する。
まずは定義したモデルのインスタンスを作成。

```python
model = Mynet()
```
そして肝心のoptimizerの設定ではmodel.compileという関数を使う。ここで学習率だとかモーメンタムだとか重要なハイパーパラメータを設定する。
ここではSGDで学習率0.001, モーメンタム0.9を設定。metrics=['accuracy']とすると自動的にaccuracyも計算してくれる。

```python
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
    metrics=['accuracy'])
```

## <a id="dataset">4. データセット用意</a>

あとは学習させるだけなのでデータセットを用意する。一応再掲。詳しくはディープラーニング準備編を要参照。

```bash
# get train data
def data_load(path, hf=False, vf=False):
    xs = np.ndarray((0, img_height, img_width, 3), dtype=np.float32)
    ts = np.ndarray((0, num_classes))
    paths = []

    for dir_path in glob(path + '/*'):
        for path in glob(dir_path + '/*'):
            x = cv2.imread(path)
            x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
            x /= 255.
            xs = np.r_[xs, x[None, ...]]

            t = np.zeros((num_classes))
            if 'akahara' in path:
                t[0] = 1
            elif 'madara' in path:
                t[1] = 1
            t = t[None, ...]
            ts = np.r_[ts, t]

            paths.append(path)

            if hf:
                _x = x[:, ::-1]
                xs = np.r_[xs, _x[None, ...]]
                ts = np.r_[ts, t]
                paths.append(path)

            if vf:
                _x = x[::-1]
                xs = np.r_[xs, _x[None, ...]]
                ts = np.r_[ts, t]
                paths.append(path)

            if hf and vf:
                _x = x[::-1, ::-1]
                xs = np.r_[xs, _x[None, ...]]
                ts = np.r_[ts, t]
                paths.append(path)

    return xs, ts, paths

xs, ts, paths = data_load('../Dataset/train/images/', hf=True, vf=True)
```

## <a id="train">5. 学習</a>

kerasの学習はfitなどの関数があるがここではあえて使わない。
ここからミニバッチを使って学習させる。100イテレーションを想定して、こんな感じでミニバッチを作成する。ミニバッチの作成の詳細はディープラーニング準備編を要参照。これで、xとtに学習データの入力画像、教師ラベルが格納される。

```python
mb = 8
mbi = 0
train_ind = np.arange(len(xs))
np.random.seed(0)
np.random.shuffle(train_ind)

for i in range(100):
    if mbi + mb > len(xs):
        mb_ind = train_ind[mbi:]
        np.random.shuffle(train_ind)
        mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
        mbi = mb - (len(xs) - mbi)
    else:
        mb_ind = train_ind[mbi: mbi+mb]
        mbi += mb

    x = xs[mb_ind]
    t = ts[mb_ind]
```

学習では*train_on_batch*というメソッドで行える。返し値はlossとaccuracyになっている。


```python
for i in range(100):
    # syoryaku ...
    
    loss, acc = model.train_on_batch(x=x, y=t)
    print("iter >>", i+1, ",loss >>", loss, ',accuracy >>', acc)
```

## <a id="save">6. 学習済みモデルの保存</a>

モデルを学習したらそのパラメータを保存しなきゃいけない。それは*model.save()*を使う。保存名は*model.h5*とする。

```python
for i in range(100):
    # syorayku ...
        
model.save('model.h5')
```

以上で学習が終了!!

## <a id="test">7. 学習済みモデルでテスト</a>

次に学習したモデルを使ってテスト画像でテストする。

モデルの準備らへんはこんな感じ。*model.load_weights()* で学習済みモデルを読み込める。

```python
model = Mynet()
model.load_weights('model.h5')
````

あとはテストデータセットを読み込む。

```python
xs, ts, paths = data_load('../Dataset/test/images/')
```

あとはテスト画像を一枚ずつモデルにフィードフォワードして予測ラベルを求めていく。これは*predict_on_batch*を使う。

```python
for i in range(len(paths)):
    x = xs[i]
    t = ts[i]
    path = paths[i]
    
    x = np.expand_dims(x, axis=0)

    pred = model.predict_on_batch(x)[0]
    print("in {}, predicted probabilities >> {}".format(path, pred))
```

以上でtensorflowの使い方は一通り終了です。お互いおつです。


## <a id="code">8. まとめたコード<a>

以上をまとめたコードは *main_keras.py*　です。使いやすさのために少し整形してます。

学習は

```bash
$ python main_keras.py --train
```

テストは

```bash
$ python main_keras.py --test
```

## <a id="api">KerasのAPIを使う</a>

こんなディレクトリ構成にする

```python
here --- Dataset --- Train --- class1 --- *.jpg
                  |         |- class2 --- *.jpg
                  |
                  |- Val  --- class1 --- *.jpg
                  |         |- class2 --- *.jpg
                  |
                  |- Test  --- class1 --- *.jpg
                            |- class2 --- *.jpg
```

コードはこんな感じ。KerasではDataGeneratorというAPIが用意されている。

```python
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

import cv2
from glob import glob

# GPU config
import tensorflow as tf
from keras import backend as K

config = tf.ConfigProto()
# keep GPU memory to use
config.gpu_options.allow_growth = True
# GPU device number
config.gpu_options.visible_device_list="0"
sess = tf.Session(config=config)
K.set_session(sess)

# escape warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# configure
# input shape
img_height, img_width, channel = 256, 256, 3

# dataset path
train_path = "Dataset/Train"
val_path = "Dataset/Val"
test_path = "Dataset/Test"

# model save path
model_path = "model.h5"

# model definition
def Res50():
    # define backbone
    from keras.applications.resnet50 import ResNet50
    model = ResNet50(include_top=False,  # do not contain last MLP layers
        weights='imagenet',   # use imagenet pretrained model
        input_shape=(img_height, img_width, channel)  # input shape
        )

    # get model
    last = model.output
    # flatten [mb, h, w, c] -> [mb, c]
    x = Flatten()(last)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(cf.Class_num, activation='softmax')(x)
    model = Model(model.input, x)
    return model


# train script
def train():
    # model
    model = Res50()
    
    # set optimizer
    model.compile(loss='categorical_crossentropy',
        optimizer=keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
        metrics=['accuracy'])
      
    # preprocess for Data generator
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        )
        
    val_datagen = ImageDataGenerator(
        rescale=1. / 255,
        )
    
    # make Data generator
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(img_height, img_width),
        batch_size=30,
        class_mode='categorical')
        
    val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=(img_height, img_width),
        batch_size=1,
        class_mode='categorical')
        
    # train
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=50,
        epochs=100,
        validation_data=val_generator,
        validation_steps=50
    )
    
    # save trained model
    model.save(model_path)
    
    
# test
def test():
    # model
    model = Res50()
    
    # load trained parameters
    model.load_weights(model_path)
    
    # get test data paths
    test_dir_paths = glob(test_path + "/*")
    
    # get class directory path
    for test_dir_path in test_dir_paths:
        # get image paths
        img_paths = glob(test_dir_path)
        
        for img_path in img_paths:
            # get image and preprocess
            img = cv2.imread(img_path)
            img = img[..., ::-1]
            img = img.astype(np.float32) / 255.
            img = np.expand_dims(img, axis=0)
            
            # predict
            y = model.predict_on_batch(img)[0]
            
            # get score
            score = y.max()
            
            # get predict index
            pred_ind = y.argmax()
            
train()
test()
```


## <a id="error">エラー集</a>

#### 1. WARNING: AVX2 FMA

1. エラー内容

```bash
I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
```

2. 解決法

以下をpythonファイルの先頭に記述

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

#### 2. メモリ解放

```python
from keras import backend as K
K.clear_session()
tf.reset_default_graph()
```
