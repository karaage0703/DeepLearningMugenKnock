# PyTorchを使ったった

- [インポート](#import)
- [モデル](#model)
- [GPU設定](#gpu)
- [最適化の設定](#optimize)
- [学習データセットの準備](#dataset)
- [学習](#train)
- [学習済みモデルの保存](#save)
- [テスト](#test)
- [コード](#code)

## <a id="import">1. インポート</a>

最初は必要なものをpipでインストールする。

```bash
$ pip install torch torchvision argparse opencv-python numpy glob
```

コードではtorchとtorch.nn.functionalをimport する。
torchはいろいろな大事な関数が入っている。
torch.nn.functional は活性化関数relu, softmaxとかがメインで入っている。Fとエイリアスをつけられることが多い。

```python
improt torch
import torch.nn.funcitonal as F
```

あとは必要なものももろもろimportする。

```python
import argparse
import cv2
import numpy as np
from glob import glob
```

次に諸々必要な宣言をする。
num_classesは分類するクラス数。今回はアカハライモリ(akahara)とマダライモリ(madara)の２クラス。
img_height, img_widthは入力する画像のサイズ。

```python
num_classes = 2
img_height, img_width = 64, 64
GPU = False
torch.manual_seed(0)
```

## <a id="model">2. モデル定義</a>

モデル定義はtorch.nn.Moduleのラッパーとしてクラスで定義する。initには学習するパラメータが必要なlayerを書く。forwardには実際のネットワークの流れを書く。forwardの最後はsoftmaxを適用していないが、これは実際の学習の時にこうする理由がある。
ここではimg_height, img_widthで入力画像のサイズを設定している。

```python
class Mynet(torch.nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        self.conv1_1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1_1 = torch.nn.BatchNorm2d(32)
        self.conv1_2 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = torch.nn.BatchNorm2d(32)
        self.conv2_1 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = torch.nn.BatchNorm2d(64)
        self.conv2_2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = torch.nn.BatchNorm2d(64)
        self.conv3_1 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = torch.nn.BatchNorm2d(128)
        self.conv3_2 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = torch.nn.BatchNorm2d(128)
        self.conv4_1 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4_1 = torch.nn.BatchNorm2d(256)
        self.conv4_2 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4_2 = torch.nn.BatchNorm2d(256)
        self.fc1 = torch.nn.Linear(img_height//16 * img_width//16 * 256, 512)
        #self.fc1_d = torch.nn.Dropout2d()
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc_out = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, img_height//16 * img_width // 16 * 256)
        x = F.relu(self.fc1(x))
        #x = self.fc1_d(x)
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        return x
```

## <a id="gpu">3. GPUの設定</a>

pytorchはGPUを使うかをコードで制御する。これで使うデバイスをGPUかCPUかを決定できる。

```python
GPU = True
device = torch.device("cuda" if GPU else "cpu")
```

## <a id="optimize">4. Optimizerの設定</a>

モデルを書いたら次に最適化optimizerを設定する。
まずは定義したモデルのインスタンスを作成。

```python
model = Mynet()
```

次にモデルのパラメータとかをGPUにおくかCPUに置くかを定義。

```python
model = model.to(device)
```

Pytorchではパラメータの更新を行うために次の一文が必要。これで学習によってパラメータが更新されるようになる。ちなみにただテストしたいときはmodel.eval()にしてパラメータが更新されないように設定しなければいけない。

```python
model.train()
```

そして肝心のoptimizerの設定。ここで学習率だとかモーメンタムだとか重要なハイパーパラメータを設定する。
ここではSGD(確率的勾配降下法)で学習率0.01, モーメンタム0.9を設定。

```python
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

## <a id="dataset">5. データセット用意</a>

あとは学習させるだけなのでデータセットを用意する。一応再掲。詳しくはディープラーニング準備編を要参照。

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

## <a id="train">6. 学習</a>

ここからミニバッチを使って学習させる。100イテレーションを想定して、こんな感じでミニバッチを作成する。ミニバッチの作成の詳細はディープラーニング準備編を要参照。これで、xとtに学習データの入力画像、教師ラベルが格納される。

```python
for i in range(100):
    if mbi + mb > len(xs):
        mb_ind = train_ind[mbi:]
        np.random.shuffle(train_ind)
        mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
    else:
        mb_ind = train_ind[mbi: mbi+mb]
        mbi += mb
        
    x = xs[mb_ind]
    t = ts[mb_ind]
```

**ただし、PyTorchはnumpyをそのままネットワークに突っ込むことができない。一度torch.tensor型にしなければいけない!!**
ということで、こんな感じで型変換する。
**教師ラベルはlong型にしないとエラー吐くので注意!!**

```python
for i in range(100):
    # syoryaku ...

    x = xs[mb_ind]
    t = ts[mb_ind]
    
    x = torch.tensor(x, dtype=torch.float).to(device)
    t = torch.tensor(t, dtype=torch.long).to(device)
```

入力画像をネットワークに入れてフィードフォワードするには、モデルの変数を関数っぽく呼べばいい。ここでフィードフォワードする前にopt.zero_grad()してoptimizerの勾配を0にしておく。

```python
for i in range(100):
    # syoryaku ...

    x = torch.tensor(x, dtype=torch.float).to(device)
    t = torch.tensor(t, dtype=torch.long).to(device)
    
    opt.zero_grad()
    y = model(x)
```

これで出力が出るわけだけど、教師ラベルとの誤差が計算されてない。そのためにネットワークの出力のsoftmaxのlogをとって、cross-entropy-lossを計算する。

```python
for i in range(100):
    # syoryaku ...
    
    y = model(x)
    y = F.log_softmax(y, dim=1)
    loss = torch.nn.CrossEntropyLoss()(y, t)
```

lossの勾配をネットワークに伝搬させるには、loss.backward()を使う。そして、opt.step()で伝搬した勾配を用いてパラメータの更新を行う。

```python
for i in range(100):
    # syoryaku ...

    loss = torch.nn.CrossEntropyLoss()(y, t)
    loss.backward()
    opt.step()
```

ここで現段階で学習データをどれくらい正確に予測できているかのAccuracyを取るには、yのargmaxを撮って、次の関数を使う。
ちなみにtorch型のままでは扱いにくいので、一度numpyにしたほうがいい。
そのときにはtorch型の変数に対して、x.dataとするとtorchからnumpyへの変換ができる。

```python
for i in range(100):
    # syoryaku ...
    
    opt.step()
    
    pred = y.argmax(dim=1, keepdim=True)
    acc = pred.eq(t.view_as(pred).sum().item() / mb)
    
    print("iter >>", i+1, "loss >>", loss.item(), "accuracy >>", acc)
```

## <a id="save">7. 学習済みモデルの保存</a>

モデルを学習したらそのパラメータを保存しなきゃいけない。それは*torch.save()*を使う。保存名は*cnn.pt*とする。
ここで、model.state_dict()としているのはモデルのパラメータのみを保存するため。これをしないとGPUを使うかなども全て保存してしまい、あとあとめんどくさくなる。

```python
torch.save(model.state_dict(), "cnn.pt")
```

以上で学習が終了!!

## <a id="test">8. 学習済みモデルでテスト</a>

次に学習したモデルを使ってテスト画像でテストする。

モデルの読み込みはこんな感じ。テスト時はmodel.eval()をしないと勝手にパラメータが更新されるので、**model.eval()は絶対不可避**。学習済みモデルを読み込むには load_state_dict()と使う。

```python
device = torch.device("cuda" if GPU else "cpu")
model = Mynet().to(device)
model.eval()
model.load_state_dict(torch.load('cnn.pt'))
````

あとはテストデータセットを読み込む。

```python
xs, ts, paths = data_load('../Dataset/test/images/')
```

あとはテスト画像を一枚ずつモデルにフィードフォワードして予測ラベルを求めていく。ただしネットワークへの入力は4次元でなければいけない。そこで、np.expand_dimsを使ってxを3次元から４次元にする必要がある。

```python
for i in range(len(paths)):
    x = xs[i]
    t = ts[i]
    path = paths[i]

    x = np.expand_dims(x, axis=0)
    x = torch.tensor(x, dtype=torch.float).to(device)

    pred = model(x)
    pred = F.softmax(pred, dim=1).detach().cpu().numpy()[0]

    print("in {}, predicted probabilities >> {}".format(path, pred))
```

以上でPyTorchの使い方は一通り終了です。おつです。


## <a id="code">9. まとめたコード</a>

以上をまとめたコードは *main_pytorch.py*　です。使いやすさのために少し整形してます。

学習は

```bash
$ python main_pytorch.py --train
```

テストは

```bash
$ python main_pytorch.py --test
```
