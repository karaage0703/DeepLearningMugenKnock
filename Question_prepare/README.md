# Q. 2ディープラーニング準備編

## Q.1. 学習データセット読み込み 

まずはディープラーニングを学習させるためのデータセットの準備をする。

フレームワークに関係なく、学習データセットの入力画像は４次元配列で準備することが殆どである。

ここでいう４次元は[データ数、画像の縦サイズ、画像の横サイズ、チャネル(RGBの3かグレースケールの1)]となる。
ただしこの順番はフレームワークによって変わる。


| フレームワーク | データの順番 |
|:---:|:---:|
| Tensorflow, Keras(Tensorflow)| [データ数、画像の縦サイズ、画像の横サイズ、チャネル] |
| PyTorch, Chainer, Caffe  | [データ数、チャネル、画像の縦サイズ、画像の横サイズ] |

つまり、変数*xs*に学習データセットが代入された場合、xs[0]とすると、1枚めの画像(Height x Width x Channelの画像データ)を取得できるようになる。

一方、教師ラベルのデータは、１次元もしくは二次元配列で用意する。
1次元の場合はクラスのインデックス(例えば３クラス分類にて犬なら0、イモリなら1、ヤモリなら2みたいな)を指定するが、二次元の場合はone-hot表現を用いる(犬なら[1,0,0]、イモリなら[0,1,0]、ヤモリなら[0,0,1]みたいな)。
これもフレームワークによって変わる。


| フレームワーク | 教師ラベルのデータ |
|:---:|:---:|
| PyTorch, Chainer | index [データ数] |
| Tensorflow, Keras(Tensorflow), Caffe  | one-hot [データ数、クラス数] |

**このようにフレームワークによって用意するデータの形が変わるので注意**

ここではdata_load()という関数を用意して、画像の学習データセット、教師データセットを用意して、それぞれxs, tsという変数に格納してxs, tsを返す使用にせよ。対象とするフレームワークは**PyTorch**で*Dataset/train/images*のアカハライモリとマダライモリの２クラス分類を想定。
用いるデータセットは*Dataset/train/images*の*akahaara*と*madara*を用いよ。
akaharaはクラス0、madaraはクラス1に属する。

プログラムの順番は、
1. xs, tsを用意。
2. ディレクトリの画像xを読み込み、固定サイズにリサイズする。値を255で割って[0,1]の範囲に正規化する。
3. xsにxを追加する。 xs = np.vstack((xs, x))
4. 教師ラベルtを何らかの方法で取得する。
5. tsにtを追加する。 ts = np.vstack((ts, t))
6. pathsに画像のパスを追加appendする
7. 2-6をディレクトリの画像全部に対して行う。

以下のコードを埋めて完成させよ。

```python
import cv2
import numpy as np
from glob import glob

num_classes = 2
img_height, img_width = 64, 64

CLS = ['akahara', 'madara']

def data_load(path):
    xs = []
    ts = []
    paths = []

    # answer your code
    ...
    #
    return xs, ts, paths

xs, ts, paths = data_load('../Dataset/train/images/')
```

答え >> [answer_data_load.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_prepare/answer_data_load.py)

## Q.2. ミニバッチ作成

学習データが完成したら、次はミニバッチを作成しなければいけない。

学習データ全部を**バッチ**という表現をしていて、**ミニバッチ**とはバッチの一部分、つまりバッチからいくつか選んだデータの集合を指す。ディープラーニングではたくさんメモリを使うので、全部のデータを一度に用いることができない。そこでバッチを分割してミニバッチにして学習させる。

ここではxs, tsからミニバッチを作成するプログラムを作成せよ。

作り方はいくつかあるがここではデータセットと同じ数のインデックスの配列を用意し、それをシャッフルしながらミニバッチを作成する。今回の学習データは16枚なので、np.arange(len(xs))で[0, 1, 2, 3, ..., 15]というインデックスの配列train_indを用意する。それをシャッフルする。ここから3つ分選んでxs[index], ts[index]とすればミニバッチを選ぶことができる。

これで選んでいくと、最後は２つしか残らなくなってしまうが、そのときはインデックス配列をシャッフルして、そこから一つ追加してミニバッチにする。

mbiに一個前に選んだインデックス配列の最後の位置を格納する。
ここではミニバッチ数mb=3を10回繰り返して選んだミニバッチのインデックスmb_indを表示せよ。

```python
mb = 3
mbi = 0
train_ind = np.arange(len(xs))
np.random.seed(0)
np.random.shuffle(train_ind)

for i in range(100):
    # answer your code
    ...
    #
    print(mb_ind)
```


答え

```bash
$ python answer_minibatch.py
[1 6 8]
[ 9 13  4]
[ 2 14 10]
[ 7 15 11]
[3 0 5]
[6 8 5]
[ 6  9 14]
[ 9 11  4]
[13 11  0]
[11  0  9]
```

答え >> [answer_minibatch.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_prepare/answer_minibatch.py)


## Q.3. イテレーションとエポック

ディープラーニングの学習はミニバッチを何回使うかの学習回数が重要となる。
これはイテレーションiterationとかstepだとかepochとかって呼ばれる。

iteration,stepは何回ミニバッチを回すか、epochはバッチを何回回すかを指すことが多い。

Q.2-2では10回ミニバッチを回したが、これは10iterationしたことになる。
ここでは3エポックを実装してみよ。


ここまでできれば、あとはディープラーニングを学習させるだけ！！

答え

```bash
[1 6 8]
[ 9 13  4]
[ 2 14 10]
[ 7 15 11]
[3 0 5]
[6 8 5]
[3 7 1]
[ 9 13 15]
[11  4 12]
[10  0 14]
[8 6 9]
[14 15  1]
[12  2 13]
[10  4  0]
[ 5 11  7]
[3 8 6]
[11  4 14]
```

答え >> [answer_epoch.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_prepare/answer_epoch.py)


## Q.4. データ拡張・左右反転

ディープラーニングは学習データがめちゃくちゃ多くないといけない。小さいモデルなら少ないデータでも大丈夫だけど、大きいモデルの方が最終的には精度は高い。しかし大きいモデルは大量のデータがないと学習が進まないというジレンマがある。これを解決するために学習データを水増しする手法はよく取られる。

ここではデータ拡張の一つとして左右を反転させる。
これはデータを読み込んだ後に、画像の左右を反転させたものも一つの学習データとして加えるものである。

*data_load()*の引数に*hf* という引数(デフォルはFalse)を加えて左右反転も加えられるようにせよ。

答え >> [answer_hf.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_prepare/answer_hf.py)

## Q.5. データ拡張・上下反転

ここではデータ拡張として上下を反転させたものを加えよ。

*data_load()*の引数に*vf* という引数(デフォルはFalse)を加えて左右反転も加えられるようにせよ。

答え >> [answer_vf.py](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_prepare/answer_vf.py)
