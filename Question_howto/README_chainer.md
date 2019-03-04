# Chainerを使ったった

## 1. インポート

最初は必要なものをpipでインストールする。

```bash
$ pip install chainer cupy argparse opencv-python numpy glob
```

GPUでtensorlfowを使う場合は、tensorflow_gpuをインストールする必要がある。

コードではchainerをimport する。
*chainer.links* は学習パラメータが必要なlayer(convolutionとか)が入っている。
*chainer.functions* は活性化関数とかが入っている。

```python
import chainer
import chainer.links as L
import chainer.functions as F
```

あとは必要なものももろもろimportする。

```python
import argparse
import cv2
import numpy as np
from glob import glob
```

次に諸々必要な宣言をする。num_classesは分類するクラス数。今回はアカハライモリ(akahara)とマダライモリ(madara)の２クラス。img_height, img_widthは入力する画像のサイズ。
GPUがマイナスのときは使わないように設定。正の値のときは使用するGPUの番号として扱う。

```python
num_classes = 2
img_height, img_width = 64, 64
GPU = -1
```


## 2. モデル定義

モデルは*chainer.Chain*を継承してクラスで定義する。
initには学習パラメータが必要なlayerを記述して、callにはネットワーク全体の構造を書く。
殆どpytorchと同じ。trainは私オリジナルで加えたもので、dropoutの挙動を学習時とテスト時で違うくしたかったので加えた。

ちなみにここではcallでsoftmaxを適用しない。これはLossの計算時にsoftmaxを適用する前のネットワークの出力が必要になるからである。

```python
class Mynet(chainer.Chain):
    def __init__(self, train=True):
        self.train = train
        super(Mynet, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(None, 32, ksize=3, pad=1, nobias=False)
            self.bn1_1 = L.BatchNormalization(32)
            self.conv1_2 = L.Convolution2D(None, 32, ksize=3, pad=1, nobias=False)
            self.bn1_2 = L.BatchNormalization(32)
            self.conv2_1 = L.Convolution2D(None, 64, ksize=3, pad=1, nobias=False)
            self.bn2_1 = L.BatchNormalization(64)
            self.conv2_2 = L.Convolution2D(None, 64, ksize=3, pad=1, nobias=False)
            self.bn2_2 = L.BatchNormalization(64)
            self.conv3_1 = L.Convolution2D(None, 128, ksize=3, pad=1, nobias=False)
            self.bn3_1 = L.BatchNormalization(128)
            self.conv3_2 = L.Convolution2D(None, 128, ksize=3, pad=1, nobias=False)
            self.bn3_2 = L.BatchNormalization(128)
            self.conv4_1 = L.Convolution2D(None, 256, ksize=3, pad=1, nobias=False)
            self.bn4_1 = L.BatchNormalization(256)
            self.conv4_2 = L.Convolution2D(None, 256, ksize=3, pad=1, nobias=False)
            self.bn4_2 = L.BatchNormalization(256)
            self.fc1 = L.Linear(None, 512, nobias=False)
            self.fc2 = L.Linear(None, 512, nobias=False)
            self.fc_out = L.Linear(None, num_classes, nobias=False)

    def __call__(self, x):
        x = F.relu(self.conv1_1(x))
        x = self.bn1_1(x)
        x = F.relu(self.conv1_2(x))
        x = self.bn1_2(x)
        x = F.max_pooling_2d(x, ksize=2, stride=2)
        x = F.relu(self.conv2_1(x))
        x = self.bn2_1(x)
        x = F.relu(self.conv2_2(x))
        x = self.bn2_2(x)
        x = F.max_pooling_2d(x, ksize=2, stride=2)
        x = F.relu(self.conv3_1(x))
        x = self.bn3_1(x)
        x = F.relu(self.conv3_2(x))
        x = self.bn3_2(x)
        x = F.max_pooling_2d(x, ksize=2, stride=2)
        x = F.relu(self.conv4_1(x))
        x = self.bn4_1(x)
        x = F.relu(self.conv4_2(x))
        x = self.bn4_2(x)
        x = F.max_pooling_2d(x, ksize=2, stride=2)
        x = F.relu(self.fc1(x))
        if self.train:
            x = F.dropout(x, ratio=0.5)
        x = F.relu(self.fc2(x))
        if self.train:
            x = F.dropout(x, ratio=0.5)
        x = self.fc_out(x)
        return x
```

## 2. GPUの設定

GPUを使うときはそれようの記述が必要。

まずはモデルのインスタンス生成。

```python
model = Mynet(train=True)
```

次にGPUを使うために、パラメータをGPUのメモリ用にラップします。

```python
if GPU >= 0:
    chainer.cuda.get_device(GPU).use()
    model.to_gpu()
```


## 3. Optimizerの設定

optimizerの設定では*chainer.optimizers* から選ぶ。ここで学習率だとかモーメンタムだとか重要なハイパーパラメータを設定する。
optimizerを定義したら、opt.setup(model)でモデルにセットする。
ここではSGDで学習率0.001, モーメンタム0.9を設定。更に重み減衰0.0005もつけてみる。

```python
opt = chainer.optimizers.MomentumSGD(lr=0.001, momentum=0.9)
opt.setup(model)
opt.add_hook(chainer.optimizer.WeightDecay(0.0005))
```

## 4. データセット用意

あとは学習させるだけなのでデータセットを用意する。一応再掲。詳しくはディープラーニング準備編を要参照。
chainerでは入力画像は*float32*、教師ラベルは*int*でなければエラーを吐くので注意！！

```bash
# get train data
def data_load(path, hf=False, vf=False):
    xs = np.ndarray((0, img_height, img_width, 3), dtype=np.float32)
    ts = np.ndarray((0), dtype=np.int)
    paths = []

    for dir_path in glob(path + '/*'):
        for path in glob(dir_path + '/*'):
            x = cv2.imread(path)
            x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
            x /= 255.
            xs = np.r_[xs, x[None, ...]]

            t = np.zeros((1))
            if 'akahara' in path:
                t = np.array((0), dtype=np.int)
            elif 'madara' in path:
                t = np.array((1), dtype=np.int)
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

    xs = xs.transpose(0,3,1,2)

    return xs, ts, paths

xs, ts, paths = data_load('../Dataset/train/images/', hf=True, vf=True)
```

## 5. 学習

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

GPUを使う場合は*chaienr.cuda.to_gpu()* の型に入れなきゃいけない。

```bash
for i in range(100):
    # syoryaku ...
 
    x = xs[mb_ind]
    t = ts[mb_ind]
    
    if GPU >= 0:
        x = chainer.cuda.to_gpu(x)
        t = chainer.cuda.to_gpu(t)
```

ネットワークへのフィードフォワードはモデルをメソッドっぽく呼べばいい。


```python
for i in range(100):
    # syoryaku ...
    
    if GPU >= 0:
        x = chainer.cuda.to_gpu(x)
        t = chainer.cuda.to_gpu(t)
        
    y = model(x)
```

Lossの計算は*F.softmax_cross_entropy()* 、Accuracyの計算は*F.accuracy()* を用いる。
lossの逆伝搬は*loss.backward()* ででき、重みの更新は *opt.update()* でできる。
*backward(), update()* する前にモデルに保存されている勾配を*model.cleargrads()* で消すことができる。


```python
for i in range(100):
    # syoryaku ...

    y = model(x)
    
    loss = F.softmax_cross_entropy(y, t)
    accu = F.accuracy(y, t)
    
    model.cleargrads()
    loss.backward()
    opt.update()
```

ちなみにloss, accuは*chainer.Variable()* というchainer独自の型に入っているため、このままでは扱いがめんどい。
numpyとして値を取るには *loss.data* としなればいけない。さらにGPUを使っているときは、*chainer.cuda* の型に入っているため、*chainer.cuda.to_cpu()* でcpuに移す必要がある。
図にするとこういう関係。

```bash
             CPU           ||             GPU
==============================================================
      x = model(x)              x = chainer.cuda.to_gpu(x)
numpy  --------->  chainer.Variable  ---------->  chainer.cuda
       <--------                     <----------
      x = x.data                x = chainer.cuda.to_cpu(x)
```

これでloss, accuracyがとれる。

```python
for i in range(100):
    # syoryaku ...

    model.cleargrads()
    loss.backward()
    opt.update()
    
    loss = loss.data
    accu = accu.data
   
    if GPU >= 0:
        loss = chainer.cuda.to_cpu(loss)
        accu = chainer.cuda.to_cpu(accu)
    
    print("iter >>", i+1, ',loss >>', loss.item(), ',accuracy >>', accu)
```


## 7. 学習済みモデルの保存

モデルを学習したらそのパラメータを保存しなきゃいけない。それは*chainer.serializers.save_npz()*を使う。保存名は*cnn.npz*とする。

```python
for i in range(100):
    # syorayku ...

chainer.serializers.save_npz('cnn.npz', model)
```

以上で学習が終了!!

## 8. 学習済みモデルでテスト

次に学習したモデルを使ってテスト画像でテストする。

モデルの準備らへんはこんな感じ。*chainer.serializers.load_npz()* で学習済みモデルを読み込める。

```python
model = Mynet(train=False)

if GPU >= 0:
    chainer.cuda.get_device_from_id(cf.GPU).use()
    model.to_gpu()

## Load pretrained parameters
chainer.serializers.load_npz('cnn.npz', model)
````

あとはテストデータセットを読み込む。

```python
xs, ts, paths = data_load('../Dataset/test/images/')
```

あとはテスト画像を一枚ずつモデルにフィードフォワードして予測ラベルを求めていく。ここではモデルの出力に*F.softmax()* を使って確率に直す。

```python
for i in range(len(paths)):
    x = xs[i]
    t = ts[i]
    path = paths[i]
    
    x = np.expand_dims(x, axis=0)

    if GPU >= 0:
        x = chainer.cuda.to_gpu(x)

    pred = model(x).data
    pred = F.softmax(pred)

    if GPU >= 0:
        pred = chainer.cuda.to_cpu(pred)

    pred = pred[0].data

    print("in {}, predicted probabilities >> {}".format(path, pred))
```

以上でtensorflowの使い方は一通り終了です。お互いおつです。


## 9. まとめたコード

以上をまとめたコードは *main_chainer.py*　です。使いやすさのために少し整形してます。

学習は

```bash
$ python main_chainer.py --train
```

テストは

```bash
$ python main_chainer.py --test
```


## エラー&警告

### Waring: Please be aware that Mac OS X is not an officially supported OS.

MacOSXでchainerをimportするとでるwarning。

まずはbrewでopenblasをインストール

```bash
$ brew install openblas
```

そしたら一度numpyをアンインストール

```bash
$ pip uninstall numpy
```

ホームに *.numpy-site.cfg* という名前のファイルを作成して、以下を書き込む。

```bash
[openblas]
libraries = openblas
library_dirs = /usr/local/opt/openblas/lib
include_dirs = /usr/local/opt/openblas/include
```

次のコマンドでnumpyをインストールする

```bash
$ pip install --no-binary :all: numpy
```
これでwarningがでなくなる。

参考 >> https://docs.chainer.org/en/stable/tips.html

