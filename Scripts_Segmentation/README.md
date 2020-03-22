# Q. SemanticSegmentation編

ここではSemanticSegmentationの実装を行っていきます。

## SemanticSegmentationとは?

これは画像に対してピクセル毎になんのクラスに属するかを予測するタスクです。

たとえばこんな感じです。赤が 「アカハライモリ」 、緑が「マダライモリ」、黒が「背景」

| 入力1(akahara_0005.jpg) | 出力1(akahara_0005.png) | 入力2(madara_0007.jpg) | 出力2(madara_0007.png) |
|:---:|:---:|:---:|:---:|
| ![](assets/akahara_0005.jpg) | ![](assets/akahara_0005.png) |  ![](assets/madara_0007.jpg) |  ![](assets/madara_0007.png) |

SemanticSegmentataionのイメージが掴めたでしょう。

これもディープラーニングでできるので、ここから実装していきます。

ここでも、**とりあえずPyTorch, Tensorflow, Keras, Chainer全部で実装してください。**
ネットワークを作ったら、学習率やイテレーションを変えて、テストデータセット *../Dataset/test/images* でテストしてください。

**ちなみにSemaSegのGround-truthは *../Dataset/train/seg_images* にあります。
アカハライモリは RGB=(127,0,0)、マダライモリはRGB=(0,127,0)、背景はRGB=(0,0,0)で設定してます。**

## Binalization Step.1. データセット読み込み

まずは簡単にSigmoidを使って、イモリか背景かを分類するSemanticSegmentationのモデルを作っていきます。

ここではアカハライモリとマダライモリを区別せずに、イモリか否かの２クラスとなります。が、２クラスの場合はsigmoidを使うことで、ピクセル毎にイモリの確率を出せばいいです。

つまり出力はイモリの部分の確率が高くなるようにすればいいよくて、こんなかんじ。

| 入力1(akahara_0005.jpg) | 出力1(akahara_0005_bin.png) | 入力2(madara_0007.jpg) | 出力2(madara_0007_bin.png) |
|:---:|:---:|:---:|:---:|
| ![](assets/akahara_0005.jpg) | ![](assets/akahara_0005_bin.png) |  ![](assets/madara_0007.jpg) |  ![](assets/madara_0007_bin.png) |


ということでこれを実現していきます！！

まずはデータセットの読み込みから。

入力画像の方は画像認識のときと同じで良いが、SemaSegではGround-truthの方も画像になっているので、工夫が必要です。
Groud-truthは *../Dataset/train/seg_images* にpngで入ってます。(なぜpngかというとpngにしなければ画素の値がずれることがあるから。つまりground-truthとして機能させるにはpngとか値が絶対に変わらない拡張でなければならない)

pngを読み込んだら、イモリの位置になっている部分が1、それ以外が0となるようにground-truthを作成してください。

答え
- Pytorch [scripts_pytorch/bin_dataset_pytorch.py](scripts_pytorch/bin_dataset_pytorch.py)
- Tensorflow [scripts_tf_slim/bin_dataset_tensorflow_slim.py](scripts_tf_slim/bin_dataset_tensorflow_slim.py)
- Keras [scripts_keras/bin_dataset_keras.py](scripts_keras/bin_dataset_keras.py)
- chainer [scripts_chainer/bin_dataset_chainer.py](scripts_chainer/bin_dataset_chainer.py)

## Binalization Step.2. 学習時のLoss計算

あとは学習するだけでっす。

ここでLossは*Sigmoid Cross Entropy* を用います。各フレームワーク毎の使い方はこうです。

### PyTorch
*torch.nn.BCELoss* を使います。ただし、これの引数はy, tとも*torch.float*型であるので注意。

```python
y = model(x)
y = y.permute(0,2,3,1).contiguous()
y = torch.sigmoid(y)
loss = torch.nn.BCELoss()(y, t)
```

まずはシンプルな構造として、入力画像に対して、
*Convolution(kernel_size=3, kernel_number=32, padding=1, stride=1) + ReLU + BatchNormalization()*
を6回適用して最後に *Convolution(kernel_size=1, kernel_number=1, padding=0, stride=1)* を適用するネットワークを作成し、SigmoidによるBinalizationを実現してください。入力画像サイズは64とします。（大きくしても良いがGPUを使わないと計算時間がキツキツになってしまうので注意！）(BatchNormalizationの後にReLUする場合もあるが、ReLUの後にBNの法が今回はきれいにいったのでこれでいきます)

答え
- Pytorch [scripts_pytorch/bin_loss_pytorch.py](scripts_pytorch/bin_loss_pytorch.py)
- Tensorflow [scripts_tf_slim/bin_loss_tensorflow_slim.py](scripts_tf_slim/bin_loss_tensorflow_slim.py)
- Keras [scripts_keras/bin_loss_keras.py](scripts_keras/bin_loss_keras.py)
- chainer [scripts_chainer/bin_loss_chainer.py](scripts_chainer/bin_loss_chainer.py)

## Binalization Step.3. テスト時の予測結果の表示

学習ができたら、テストあるのみ。ということで、テストデータセット *../Dataset/test/images* に対してのネットワークの出力を図示してください。

図示はシンプルに出力した結果と閾値0.5で２値化したものを表示してください。
２値化とは、あらかじめ決定した閾値以上の値を1、それ以外を0にする画像処理手法です。

だいたい500イテレーションくらい回したpytorchでの結果がこれです。画像処理ではマスク画像作成などでこのようなBinalizationが行われるので、最後に２値化することはよくあります。

| madara_0010.jpg (answers_image/answer_bin_loss_pytorch_madara_0010.png) | akahara_0011.jpg (answers_image/answer_bin_loss_pytorch_akahara_0011.png) |
|:---:|:---:|
| ![](answers_image/answer_bin_loss_pytorch_madara_0010.png) | ![](answers_image/answer_bin_loss_pytorch_akahara_0011.png) |


答え
- Pytorch [scripts_pytorch/bin_test_pytorch.py](scripts_pytorch/bin_test_pytorch.py)
- Tensorflow [scripts_tf_slim/bin_test_tensorflow_silm.py](scripts_tf_slim/bin_test_tensorflow_slim.py)
- Keras [scripts_keras/bin_test_keras.py](scripts_keras/bin_test_keras.py)
- chainer [scripts_chainer/bin_test_chainer.py](scripts_chainer/bin_test_chainer.py)

## Semantic Segmentation Step.1. データセット読み込み

ここからはBinalizationでなくクラス分類も含めたSemantic Segmentationを行っていきます。

まずはデータセットの読み込みから。それぞれのフレームワークでやり方が少しことなるので注意。

### PyTorch

出力サイズと同じサイズのゼロ行列を容易し、各ピクセル毎にクラスのインデックスを割り当てていきます。

```python
t = np.zeros((out_height, out_width), dtype=torch.int)
```

アカハライモリとマダライモリでこういう感じで、紫がインデックス0で背景、緑がインデックス1でアカハライモリ、黄色がインデックス2でマダライモリとなります。

| assets/semaseg_gt_akahara_0008.png | assets/semaseg_gt_madara_0005.png |
|:---:|:---:|
| ![](assets/semaseg_gt_akahara_0008.png) | ![](assets/semaseg_gt_madara_0005.png) |

答え
- Pytorch [scripts_pytorch/semaseg_dataset_pytorch.py](scripts_pytorch/semaseg_dataset_pytorch.py)
- Tensorflow [scripts_tf_slim/semaseg_dataset_tensorflow_slim.py](scripts_tf_slim/semaseg_dataset_tensorflow_slim.py)
- Keras [scripts_keras/semaseg_dataset_keras.py](scripts_keras/semaseg_dataset_keras.py)
- chainer [scripts_chainer/lenet_chainer.py](scripts_chainer/semaseg_dataset_chainer.py)

## Semantic Segmentation Step.2. 学習時のLoss計算

データセットを容易すればあとは学習させるだけ！ということで学習に移るけど、SemaSegの場合はBinalizationと違って、Softmaxを使う必要です。しかし、シンプルにSoftmaxを使ってはだめで、画像に対するSoftmaxのテクニックが必要です。

方法としては、**モデルの出力は[ミニバッチ、高さ、幅、クラス数]のshapeを持っているが、これを[ミニバッチx高さx幅、クラス数]にreshapeして、softmaxとcross entropyを適用します**。これにより画像に対するsoftmaxの学習が実現できます。これは

pytorchやchainerは[ミニバッチ、チャネル、高さ、幅]のshapeだが、これをpytorchなら*permute()* を使って[ミニバッチ、高さ、幅、チャネル]の順に入れ替える必要があるので注意です。

それぞれのアルゴリズム実装は次のようになります。ネットワーク構成はBinalization時と同じとします。

### Pytorch
1. 教師データを *permutate()* を用いて[ミニバッチ、高さ、幅、クラス数]の順にする。
2. 教師データを *view()* を用いて[ミニバッチx高さx幅、クラス数]にreshapeする。
3. ネットワークの出力を *view()* を用いて[ミニバッチx高さx幅]にreshapeする。
4. あとは物体認識と同じくsoftmax cross entropyを計算する。

だいたい500イテレーションくらい学習したpytorchでの結果がこれです。画像処理ではマスク画像作成などでこのようなBinalizationが行われるので、最後に２値化することはよくあります。


答え
- Pytorch [scripts_pytorch/semaseg_loss_pytorch.py](scripts_pytorch/semaseg_loss_pytorch.py)
- Tensorflow [scripts_tf_slim/semaseg_loss_tensorflow_slim.py](scripts_tf_slim/semaseg_loss_tensorflow_slim.py)
- Keras [scripts_keras/semaseg_loss_keras.py](scripts_keras/semaseg_loss_keras.py)
- chainer [scripts_chainer/lenet_chainer.py](scripts_chaienr/semaseg_loss_chainer.py)

## Semantic Segmentation Step.3. テスト時の予測結果の表示

あとはテストデータで予測結果を見るだけ。出力はミニバッチ部分を取ると[高さ、幅、クラス数]となるので、numpyのargmaxをうまいこと使うと、ピクセル毎の確率が最大のクラスのインデックスを取ることができます。

それを用いて、アカハライモリのクラスはRGB=(127,0,0)、マダライモリはRGB=(0,127,0)、背景はRGB=(0,0,0)になるようなセグメンテーション画像を表示してください。

これはPyTorchの出力例。

| answer_semaseg_loss_pytorch_madara_0010.png | answer_semaseg_loss_pytorch_akahara_0011.png |
|:---:|:---:|
| ![](answers_image/answer_semaseg_loss_pytorch_madara_0010.png) | ![](answers_image/answer_semaseg_loss_pytorch_akahara_0011.png) |

答え
- Pytorch [scripts_pytorch/semaseg_test_pytorch.py](scripts_pytorch/semaseg_loss_pytorch.py)
- Tensorflow [scripts_tf_slim/semaseg_test_tensorflow_slim.py](scripts_slim/semaseg_test_tensorflow_slim.py)
- Keras [scripts_keras/semaseg_test_keras.py](scripts_keras/semaseg_test_keras.py)
- chainer [scripts_chainer/lenet_chainer.py](scripts_chainer/semaseg_test_chainer.py)

## UpSampling手法1. NearestNeighbor補間

ここまでで、学習の方法はができたので、あとは論文に沿って忠実にネットワークを再現していきます。

CNNはよくMaxPoolingやstride2のconvolutionなどで画像サイズを小さくしていくので(Down sampling、ダウンサンプリングという)、入力画像のサイズと比べると段々小さくなってしまいます。しかしセグメンテーションではサイズが小さいと解像度が小さくなってしまい、きれいな出力が得られません。なので、画像サイズを上げるためにUpSampling（アップサンプリング）する必要があります。

よくあるのは、NearestNeighbor補間(最近傍補間)でサイズを２倍にする方法です。この補間は画像処理でよく使われる手法です。一度MaxpoolingしてNearestNeighborでupsampleしていきます。

ネットワーク構成は、
1. Convoluton(kernel_size=3, kerner_num=32, padding=1, stride=1) + ReLU + BN を２回
2. MaxPooling(kernel_size=2, stride=2)
3. Convoluton(kernel_size=3, kerner_num=32, padding=1, stride=1) + ReLU + BN を２回
4. Bilinear補間でサイズを２倍にする。
5. Convoluton(kernel_size=3, kerner_num=32, padding=1, stride=1) + ReLU + BN を２回
6. Convoluton(kernel_size=1, kerner_num=3, padding=0, stride=1) + Softmax
としてsegmentationを実現してください。

Nearest Neighborの方法

| FW | function | | FW | function |
|:---:|:---:|:---:|:---:|:---:|
| pytorch | torch.nn.functional.interpolate | | Keras | keras.layers.UpSampling2D |
| TensorFlow | tf.image.resize_images | | Chainer | chainer.functions.resize_image (NearestNeighborがなかったのでBilinearで対応) |

これはPyTorchの出力例。

| answer_nearest_pytorch_madara_0010.png | answer_nearest_pytorch_akahara_0011.png |
|:---:|:---:|
| ![](answers_image/answer_nearest_pytorch_madara_0010.png) | ![](answers_image/answer_nearest_pytorch_akahara_0011.png) |

答え
- Pytorch [scripts_pytorch/nearest_pytorch.py](scripts_pytorch/nearest_pytorch.py)
- Tensorflow [scripts_tf_slim/nearest_tensorflow_slim.py](scripts_tf_slim/nearest_tensorflow_slim.py)
- Keras [scripts_keras/nearest_keras.py](scripts_keras/nearest_keras.py)
- chainer [scripts_chainer/nearest_chainer.py](scripts_chainer/nearest_chainer.py)

## UpSampling手法2. Transposed Convolution

２つ目の方法はTransposed convolutionを使う方法です。これはconvolutionの逆なんですが、たまに*deconvolution* と言われたりするんですが、その呼び方は正しくないらしく、*transposed convolution* の方がいいらしいです。（たしかDCGANの論文に書いてました。）

通常のconvolutionはstride=2にすると、画像サイズが半分になるのですが、**transposed convはstride=2にすると、画像サイズが２倍になります。**　これはtranspose convでは入力画像の１列毎、一行ごとに0のパディングを加えていくからです。パディング後にconvolutionのようにカーネル処理をstride=1にして行うので、画像サイズが2倍になります。つまり、transposed convのstrideとは0パディングの大きさを意味するのです。

個人的体感だけど、Transposed Convolutionの方が学習の収束が悪い気がします。多分パラメータが増えていることで学習が難しくなっているので、NearestNeighborを使った方がいいと思いますマル

ネットワークのnearest neighborの部分 Transposed conv(kernel_size=2, kernel_num=32, stride=2) + ReLU + BN に変更してネットワークを作成してください。

Transposed convolutionの方法

| FW | function | | FW | function |
|:---:|:---:|:---:|:---:|:---:|
| pytorch | torch.nn.ConvTranspose2d | | Keras | keras.layers.Conv2DTranspose |
| TensorFlow | slim.conv2d_transpose | | Chainer | chainer.links.Deconvolution2D (なぜかchainerはdeconvolutionの名前) |

| answer_transposeconv_pytorch_madara_0010.png | answer_transposeconv_pytorch_akahara_0011.png |
|:---:|:---:|
| ![](answers_image/answer_transposeconv_pytorch_madara_0010.png) | ![](answers_image/answer_transposeconv_pytorch_akahara_0011.png) |

答え
- Pytorch [scripts_pytorch/transposeconv_pytorch.py](scripts_pytorch/transposeconv_pytorch.py)
- Tensorflow [scripts_tf_slim/transposeconv_tensorflow_slim.py](scripts_tf_slim/transposeconv_tensorflow_slim.py)
- Keras [scripts_keras/transposeconv_keras.py](scripts_keras/transposeconv_keras.py)
- chainer [scripts_chainer/transposeconv_chainer.py](scripts_chainer/transposeconv_chainer.py)

ここまででセグメンテーションを行うための基本は終わり。次から論文の実装に入っていきます！！

## 特徴マップのconcat

ここではセグメンテーションのもう一つのテクニックを紹介します。それが特徴マップの結合(concat)です。よくあるのが、一度downsampleしてupsapmleしたものとdownsampleする前の特徴マップをチャンネル方向に結合する方法です。これによって様々な特徴量を利用できるとされていて、また、プーリングによって荒くなった特徴量の活性化を精細にできるとも考えられている。

![](assets/concat.png)

このように少し前の層の特徴マップをチャネル方向に結合する手法は **skip connection** と呼ばれ、ResNetで提案されたものである。これによって出力層から伝搬してきたlossの勾配を減少せずに伝搬できると考えられている。

ここではTransposed convolutionで作成したネットワークでの特徴マップのconcatを追加せよ。特徴マップを合成したものに対しては、convolution(kernel_size=1, kernel_num=結合前のチャネル, padding=0, stride=1)を適用することが多い。これにより、特徴マップの結合によりチャネル数が多くなり計算量が増えることを防ぎ、特徴量を合成することができる。

1. Convoluton(kernel_size=3, kerner_num=32, padding=1, stride=1) + ReLU + BN を２回
2. MaxPooling(kernel_size=2, stride=2)
3. Convoluton(kernel_size=3, kerner_num=32, padding=1, stride=1) + ReLU + BN を２回
4. Transposed convolution(kernel_size=2, kernel_num=32, padding=0, stride=2) + ReLU + BN
5. 4と1をチャネル方向に結合
6. Convolution(kernel_size=1, kernel_size=32, padding=0, stride=1) + ReLU + BN
6. Convoluton(kernel_size=3, kerner_num=32, padding=1, stride=1) + ReLU + BN を２回
7. Convoluton(kernel_size=1, kerner_num=3, padding=0, stride=1) + Softmax

結合の方法

| FW | function | | FW | function |
|:---:|:---:|:---:|:---:|:---:|
| pytorch | torch.cat | | Keras | keras.layers.concatenate |
| TensorFlow | tf.concat | | Chainer | chainer.links.Deconvolution2D (なぜかchainerはdeconvolutionの名前) |

答え
- Pytorch [scripts_pytorch/concat_pytorch.py](scripts_pytorch/concat_pytorch.py)
- Tensorflow [scripts_tf_slim/concat_tensorflow_slim.py](scripts_tf_slim/concat_tensorflow_slim.py)
- Keras [scripts_keras/concat_keras.py](scripts_keras/concat_keras.py)
- chainer [scripts_chainer/concat_chainer.py](scripts_chainer/concat_chainer.py)

## UNet

論文 
- U-Net https://arxiv.org/abs/1505.04597 (2015)

まずはUNet。これはもともと医療の画像処理の中で、細胞をセグメンテーションするために提案されたネットワークです。Uの由来は論文中のFig.1のモデルの形がUに見えるから。（Vやんとか言っちゃだめ）

セグメンテーションのディープラーニングでは**Encoder・エンコーダー**と**Decoder・デコーダー**という概念を持っています。
EncoderはConvolutionやPoolingによってダウンサンプリングする操作で、DecoderはTransposed convollutionやNearest Neighborによるアップサンプリングを行います。

![](assets/encoder_decoder.png)

基本的にはEncoderではConvolutionを２回したらMaxPoolingでダウンサンプリングを5回繰り返し、その後DecoderではTransposed convolutionをして同じサイズのencoderの出力をconcatしてConvolutionを２回行うのを4回繰り返します。

ただし、Convolutionはなぜかパディングなしでやっているので、画像サイズが段々小さくなっています（謎）。だからconcatする時も小さいほうのサイズに合わせて画像を切り取る（crop）する必要があります。

ネットワーク構成は
1. Encoder1: Convolution(k_size=3, k_num=64, padding=0, stride=1) + BN + ReLU を2回、 MaxPooling(k_size=2, stride=2)
2. Encoder2: Convolution(k_size=3, k_num=128, padding=0, stride=1) + BN + ReLU を2回、 MaxPooling(k_size=2, stride=2)
3. Encoder3: Convolution(k_size=3, k_num=256, padding=0, stride=1) + BN + ReLU を2回、 MaxPooling(k_size=2, stride=2)
4. Encoder4: Convolution(k_size=3, k_num=512, padding=0, stride=1) + BN + ReLU を2回、 MaxPooling(k_size=2, stride=2)
5. Encoder5: Convolution(k_size=3, k_num=1024, padding=0, stride=1) + BN + ReLU を2回
6. Transposed Convolution(K_size=2, k_num=512, padding=0, stride=2) + BN + ReLU、Encoder4の出力とconcat
7. Decoder4: Convolution(k_size=3, k_num=512, padding=0, stride=1) + BN + ReLU を2回
8. Transposed Convolution(K_size=2, k_num=256, padding=0, stride=2) + BN + ReLU、Encoder3の出力とconcat
9. Decoder3: Convolution(k_size=3, k_num=256, padding=0, stride=1) + BN + ReLU を2回
10. Transposed Convolution(K_size=2, k_num=128, padding=0, stride=2) + BN + ReLU、Encoder2の出力とconcat
11. Decoder2: Convolution(k_size=3, k_num=128, padding=0, stride=1) + BN + ReLU を2回
12. Transposed Convolution(K_size=2, k_num=64, padding=0, stride=2) + BN + ReLU、Encoder1の出力とconcat
13. Decoder1: Convolution(k_size=3, k_num=64, padding=0, stride=1) + BN + ReLU を2回
14. Convolution(k_size=1, k_num=3(クラス数), padding=0, stride=1) + Softmax
である。元論文では最後はbinalizatoinであるが、ここではSemantic Segmentationとする。入力画像のサイズは572、出力サイズは388であるが、これではGPUがないと計算時間がかかる。入力サイズを236、出力サイズ52にすればCPUでもまあまあ計算できるが、それでも遅いので、**GPUが使えない人はネットワーク作成とイテレーション１回ができればよしとします。CPUの人は次の「UNet風モデル」に移ってください。**

各FWでのクラッピングの例は以下の通り。

### Pytorch

```python
def crop_layer(layer, size):
    _, _, h, w = layer.size()
    _, _, _h, _w = size
    ph = int((h - _h) / 2)
    pw = int((w - _w) / 2)
    return layer[:, :, ph:ph+_h, pw:pw+_w]
    
_x = crop_layer(x_enc3, x.size())
```

### Tensorflow

```python
def crop_layer(layer, size):
   _, h, w, _ = layer.get_shape().as_list()
   _, _h, _w, _ = size
   ph = int((h - _h) / 2)
   pw = int((w - _w) / 2)
   return layer[:, ph:ph+_h, pw:pw+_w]
   
_enc3 = crop_layer(enc3, dec3.get_shape().as_list())
```

### Keras
```python
def crop_layer(layer, size):
    _, h, w, _ = keras.backend.int_shape(layer)
    _, _h, _w, _ = size
    ph = int((h - _h) / 2)
    pw = int((w - _w) / 2)
    return keras.layers.Cropping2D(cropping=((ph, ph), (pw, pw)))(layer)
    
_enc3 = crop_layer(enc3, keras.backend.int_shape(dec3))
```

### Chainer
```python
def crop_layer(layer, size):
    _, _, h, w = layer.shape
    _, _, _h, _w = size
    ph = int((h - _h) / 2)
    pw = int((w - _w) / 2)
    return layer[:, :, ph:ph+_h, pw:pw+_w]
    
_enc3 = crop_layer(enc3, dec3.shape)
```

答え
- Pytorch [scripts_pytorch/unet_pytorch.py](scripts_pytorch/unet_pytorch.py)
- Tensorflow [scripts_tf_slim/unet_tensorflow_slim.py](scripts_tf_slim/unet_tensorflow_slim.py)
- Keras [scripts_keras/unet_keras.py](scripts_keras/unet_keras.py)
- chainer [scripts_chainer/unet_chainer.py](scripts_chainer/unet_chainer.py)

## UNet風モデル

論文 
- U-Net https://arxiv.org/abs/1505.04597 (2015)

これは論文などはないが、UNet風のモデルを作成する。UNetではconvolutionのpaddingが0だったことなどがあり、正直使いにくく、また入力サイズから出力サイズを自分で計算しなければいけない、というしがらみがありました。そこで、ネットワーク構成をなるべくUNetにしながら、SemaSegを行うモデルを作成します。

1. Encoder1: Convolution(k_size=3, k_num=16, padding=1, stride=1) + BN + ReLU を2回、 MaxPooling(k_size=2, stride=2)
2. Encoder2: Convolution(k_size=3, k_num=32, padding=1, stride=1) + BN + ReLU を2回、 MaxPooling(k_size=2, stride=2)
3. Encoder3: Convolution(k_size=3, k_num=64, padding=1, stride=1) + BN + ReLU を2回、 MaxPooling(k_size=2, stride=2)
4. Encoder4: Convolution(k_size=3, k_num=128, padding=1, stride=1) + BN + ReLU を2回、 MaxPooling(k_size=2, stride=2)
5. Encoder5: Convolution(k_size=3, k_num=256, padding=1, stride=1) + BN + ReLU を2回
6. Transposed Convolution(K_size=2, k_num=128, padding=0, stride=2) + BN + ReLU、Encoder4の出力とconcat
7. Decoder4: Convolution(k_size=3, k_num=128, padding=1, stride=1) + BN + ReLU を2回
8. Transposed Convolution(K_size=2, k_num=64, padding=0, stride=2) + BN + ReLU、Encoder3の出力とconcat
9. Decoder3: Convolution(k_size=3, k_num=64, padding=1, stride=1) + BN + ReLU を2回
10. Transposed Convolution(K_size=2, k_num=32, padding=0, stride=2) + BN + ReLU、Encoder2の出力とconcat
11. Decoder2: Convolution(k_size=3, k_num=32, padding=1, stride=1) + BN + ReLU を2回
12. Transposed Convolution(K_size=2, k_num=16, padding=0, stride=2) + BN + ReLU、Encoder1の出力とconcat
13. Decoder1: Convolution(k_size=3, k_num=16, padding=1, stride=1) + BN + ReLU を2回
14. Convolution(k_size=1, k_num=3(クラス数), padding=0, stride=1) + Softmax

UNetからの変更点は２つ。１つはconvolutionのpaddingを1にしたこと、もう１つはカーネル数を1/4にしたこと。これなら入力サイズと出力サイズが同じになるので、ある程度使いやすくなったはずです。このように自分で論文の実装から少し変えることも重要です。入力サイズを64にしてやってみましょう。学習率などは自分でいろいろ試してきれいにセグメンテーションできるハイパーパラメータを探してみてください。

| answer_unetlike_pytorch_madara_0010.png | answer_unertlike_pytorch_akahara_0011.png |
|:---:|:---:|
| ![](answers_image/answer_unetlike_pytorch_madara_0010.png) | ![](answers_image/answer_unetlike_pytorch_akahara_0011.png) |

答え
- Pytorch [scripts_pytorch/unetlike_pytorch.py](scripts_pytorch/unetlike_pytorch.py)
-  Tensorflow [scripts_tf_slim/unetlike_tensorflow_slim.py](scripts_tf_slim/unetlike_tensorflow_slim.py)
- Keras [scripts_keras/unetlike_keras.py](scripts_keras/unetlike_keras.py)
- chainer [scripts_chainer/unetlike_chainer.py](scripts_chainer/unetlike_chainer.py)

## SegNet

元論文  
- SegNet https://arxiv.org/abs/1511.00561 (2015)

UNetではConvolutionを２層を１ブロックとして扱っていたが、SegNetではVGG16のConvolutionの構造をEncoderとして使っている。

Decoderでも対応するEncoderと同じConvolutionのBlockを使っている。

<img src="assets/segnet.png" width=600>

SegNetのDecoderにおけるアップサンプルでは、EncoderのPoolingの最大値として選ばれたインデックスを使う。

UnpoolingはPooling操作の逆で、Max poolingで選ばれたピクセルの位置と同じ位置に値を当てはめて、特徴マップを拡大する手法である。当てはめ以外のピクセルは0になる。

SegNetでは同じ特徴マップのサイズのpoolingのインデックス情報と使って、Unpoolingを行っている。

0が多くスパースに見えるが、SegNetではUnpoolingの後にConvolutionブロックを入れて、ヒスパースな特徴マップに変換している。

<img src="assets/unpooling.png" width=400>


答え
- Pytorch [scripts_pytorch/SegNet_pytorch.py](scripts_pytorch/SegNet_pytorch.py)