# Q. 画像認識編

ここではCNNの有名モデルを自分の手で実装していきます。フレームワークは自由だが、**とりあえずPyTorch, Tensorflow, Keras, Chainer全部で実装してください。**
ネットワークを作ったら、学習率やイテレーションを変えて、テストデータセット *../Dataset/test/images* でテストしてみてください。

## API

各有名モデルのAPIを使ってみましょう。

pytorchは 

```bash
$ pip install cnn_finetune
```
をして、ライブラリを追加しなきゃいけません。

| FW | function | | FW | function |
|:---:|:---:|:---:|:---:|:---:|
| pytorch | cnn_finetune.make_model  | | Keras |  |
| TensorFlow |  | | Chainer |  |

pytorchの参考サイト >> https://github.com/creafz/pytorch-cnn-finetune

答え
- Pytorch [scripts_pytorch/api_pytorch.py](scripts_pytorch/api_pytorch.py)

## Q. LeNet

元論文 >> Gradient-based learning applied to document recognition http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf (1998)

これが原初のモデル。MNISTと呼ばれる0から9までの手書き数字の判別で使われたCNNモデル。これを実装してください。LeNetはMNIST用に入力サイズが32x32となっているが、ここで用意しているデータセットは128x128サイズです。**よって学習時のデータの入力サイズを32x32にリサイズする必要があります。**

構造はこうです。

| Layer | カーネルサイズ | フィルタ数 |  ストライド| パディング |  活性化関数 | 
|:---:|:---:|:---:|:---:|:---:|:---:|
| Input | 32 x 32 x 3(入力サイズ) |
| Convolution | 5 x 5 |  6 | 1 | 0 | - |
| MaxPooling | 2 x 2 | - | 2 | 0 | sigmoid |
| Convolution | 5 x 5 | 16 | 1 | 0 | - |
| MaxPooling | 2 x 2 | - | 2 | 0 | sigmoid |
| MultiLayerPerceptron | 120 | - | - | - | - | - |
| MultiLayerPerceptron |  64 | - | - | - | - | - |
| MultiLayerPerceptron | 2 (クラス) | - | - | - | - | Softmax|



答え
- Pytorch [scripts_pytorch/lenet_pytorch.py](scripts_pytorch/lenet_pytorch.py)
- Tensorflow [scripts_tf_layers/LeNet_tf_layers.py](scripts_tf_layers/LeNet_tf_layers.py)
- Keras [scripts_keras/lenet_keras.py](scripts_keras/lenet_keras.py)
- chainer [scripts_chainer/lenet_chainer.py](scripts_chainer/lenet_chainer.py)

## Q. AlexNet

元論文 >> ImageNet Classification with Deep Convolutional Neural Networks https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks (2012)

ディープラーニングを流行らせた張本人モデル。ImageNetという画像認識のコンペILSVRC2012で圧倒的一位で優勝したことから現在のディープラーニングブームが起こりました。これを実装してくさい。
AlexNetでは*Local Response Normalization* という特別な正規化Layerがあります。

| FW | function | | FW | function |
|:---:|:---:|:---:|:---:|:---:|
| PyTorch | torch.nn.modules.normalization.LocalResponseNorm() | | Keras | - |
| TensorFlow | tf.nn.local_response_normalization() | | Chainer | chainer.functions.local_response_normalization() |

LRNは効果が薄いことから最近ではほとんど使われてません。こういうのもあったんだなあ程度に覚えておくといいと思います。

ただし学習データの枚数が少ないので学習が進まないので、精度を上げたいときは自分で学習データを増やすか、パラメータ数を変えるなどの工夫が必要なので注意。

| Layer | カーネルサイズ | フィルタ数 | ストライド| パディング | 活性化関数 | 
|:---:|:---:|:---:|:---:|:---:|:---:|
| Input | 227 x 227 x 3 (入力サイズ) |
| Convolution | 11 x 11 | 96 | 4 | 0 | ReLU |
| LocalResponseNormalization | - | - | - | - | - |
| MaxPooling | 3 x 3 | 2 | 0 | - | - |
| Convolution | 5 x 5 | 256 | 1 | 1 | ReLU |
| LocalResponseNormalization | - | - | - | - | - |
| MaxPooling | 3 x 3 | 2 | 0 | - | - |
| Convolution | 3 x 3 | 384 | 1 | 1 | ReLU |
| Convolution | 3 x 3 | 384 | 1 | 1 | ReLU |
| Convolution | 3 x 3 | 256 | 1 | 1 | ReLU |
| MaxPooling | 3 x 3 | 2 | 0 | - | - |
| MultiLayerPerceptron | 4096 | - | - | - | - | ReLU + Dropout |
| MultiLayerPerceptron | 4096 | - | - | - | - | ReLU + Dropout |
| MultiLayerPerceptron | 2 (クラス) | - | - | - | - | Softmax|


答え
- Pytorch [scripts_pytorch/alexnet_pytorch.py](scripts_pytorch/alexnet_pytorch.py)
- Tensorflow [scripts_tf_layers/AlexNet_tf_layers.py](scripts_tf_layers/AlexNet_tf_layers.py)
- Keras [scripts_keras/alexnet_keras.py](scripts_keras/alexnet_keras.py)
- chainer [scripts_chainer/alexnet_chainer.py](scripts_chainer/alexnet_chainer.py)


## Q. ZFNet

元論文 >> Visualizing and Understanding Convolutional Networks https://arxiv.org/abs/1311.2901 (2013)

ILSVRC2013で一位をとったモデル。AlexNetと構造が似てます。
Alexnetの最初のconvlutionを7x7のカーネルにして、ストライドを2に変更してます。そのかわりに２つ目のconvolutionのストライドを2にしてます。こうすることで、大きなカーネルサイズによる画像の周波数取得を変えてます。論文ではCNNが画像認識を行うまでの解析を主張してｍす。

ここらへんから計算時間がめちゃくちゃ増えるので、GPU使用をおすすめ。もしくは入力サイズを112x112とか小さくすることを推奨。

| Layer | カーネルサイズ | フィルタ数 | ストライド| パディング | 活性化関数 | 
|:---:|:---:|:---:|:---:|:---:|:---:|
| Input | 224 x 224 x 3 (入力サイズ) |
| Convolution | 7 x 7 | 96 | 2 | 0 | ReLU |
| LocalResponseNormalization | - | - | - | - | - |
| MaxPooling | 3 x 3 | 2 | 0 | - | - |
| Convolution | 5 x 5 | 256 | 2 | 1 | ReLU |
| LocalResponseNormalization | - | - | - | - | - |
| MaxPooling | 3 x 3 | 2 | 0 | - | - |
| Convolution | 3 x 3 | 384 | 1 | 1 | ReLU |
| Convolution | 3 x 3 | 384 | 1 | 1 | ReLU |
| Convolution | 3 x 3 | 256 | 1 | 1 | ReLU |
| MaxPooling | 3 x 3 | 2 | 0 | - | - |
| MultiLayerPerceptron | 4096 | - | - | - | - | ReLU + Dropout |
| MultiLayerPerceptron | 4096 | - | - | - | - | ReLU + Dropout |
| MultiLayerPerceptron | 2 (クラス) | - | - | - | - | Softmax|

答え
- Pytorch [scripts_pytorch/zfnet_pytorch.py](scripts_pytorch/zfnet_pytorch.py)
- Tensorflow [scripts_tf_layers/ZFNet_tf_layers.py](scripts_tf_layers/ZFNet_tf_layers.py)
- Keras [scripts_keras/zfnet_keras.py](scripts_keras/zfnet_keras.py)
- chainer [scripts_chainer/zfnet_chainer.py](scripts_chainer/zfnet_chainer.py)

## Q. Global Average Pooling

論文 >> Network In Network https://arxiv.org/abs/1312.4400 (2013)

ここではGlobal average poolingを実装してください。これはMLPを使わないでConvolutionaだけのモデル(**FCN: Fully Convolutional Network**)でクラス分類を行うために考案されました。通常クラス分類はMLPをクラスの数だけ用意する必要があるけど、これではネットワークへの入力サイズが固定化されてしまいます。これはMLPの性質による。しかしGAPによりこれは解決されます。

GAPはConvolutionによる生成される特徴マップの内の１チャネルの全部の値のaverage値を取る操作を行います。そうすると、チャネルの数だけの値が取れます。これにSoftmax関数を適用することでクラス分類が行われます。

アルゴリズムは、
1. ネットワークのMLPを消す。
2. クラス数の**カーネル数**(カーネルサイズでないので注意カーネルサイズはよく1x1が使われる。)を持ったConvolutionを最後につける。
3. GAPを適用する。
4. Softmaxを適用する。

これでFCNの完成です！人によってはGAP後にMLPを入れる時もあるが、どちらにしても入力画像サイズが自由なモデルが作れます。GAPはNetwork in networkの論文で提唱された手法です。

今回はZFネットにGAPを適用してください。

| FW | function | FW | function |
|:---:|:---:|:---:|:---:|
| PyTorch | torch.nn.AdaptiveAvgPooling2d() + Viewでreshape | Keras | keras.layers.GlobalAveragePooling2D() |
| TensorFlow | tf.reduce_mean()を２回適用 | Chainer | chainer.functions.average() |

答え
- Pytorch [scripts_pytorch/gap_pytorch.py](scripts_pytorch/gap_pytorch.py)
- Tensorflow [scripts_tf_layers/GAP_tf_layers.py](scripts_tf_layers/GAP_tf_layers.py)
- Keras [scripts_keras/gap_keras.py](scripts_keras/gap_keras.py)
- chainer [scripts_chainer/gap_chainer.py](scripts_chainer/gap_chainer.py)


## Q. Network in network

論文 >> Network In Network https://arxiv.org/abs/1312.4400 (2013)

Network in network(NIN)は1x1のconvoutionを導入したネットワークです。

| Layer | カーネルサイズ | フィルタ数 | ストライド| パディング | 活性化関数 | 
|:---:|:---:|:---:|:---:|:---:|:---:|
| Input | 128 x 128 x 3 (入力サイズ) |
| Convolution | 5 x 5 | 192 | 1 | 2 | ReLU |
| Convolution | 1 x 1 | 160 | 1 | 0 | ReLU |
| Convolution | 1 x 1 | 96 | 1 | 0 | ReLU |
| MaxPooling | 3 x 3 | 2 | 0 | - | - |
| Dropout |
Convolution | 5 x 5 | 192 | 1 | 2 | ReLU |
| Convolution | 1 x 1 | 192 | 1 | 0 | ReLU |
| Convolution | 1 x 1 | 192 | 1 | 0 | ReLU |
| MaxPooling | 3 x 3 | 2 | 0 | - | - |
| Dropout |
Convolution | 3 x 3 | 192 | 1 | 1 | ReLU |
| Convolution | 1 x 1 | 192 | 1 | 0 | ReLU |
| Convolution | 1 x 1 | 2(クラス数) | 1 | 0 | ReLU |
| GAP | | | | | Softmax |

答え
- Pytorch [scripts_pytorch/nin_pytorch.py](scripts_pytorch/nin_pytorch.py)
- Tensorflow [scripts_tf_layers/NIN_tf_layers.py](scripts_tf_layers/NIN_tf_layers.py)
- Keras [scripts_keras/nin_keras.py](scripts_keras/vgg16_keras.py)
- chainer [scripts_chainer/nin_chainer.py](scripts_chainer/nin_chainer.py)

## Q. VGG16

元論文 >> Very Deep Convolutional Networks for Large-Scale Image Recognition https://arxiv.org/abs/1409.1556 (2014)

VGG16とはOxfort大学の研究グループが提案したモデルであり、けっこう色んな手法のベースに使われているモデルです。VGG16では3x3のカーネルを持ったConvoutionを重ねることでモデルが取得する特徴の非線形性を増大させてます。16というのはconvolutionとMLPを合わせて16という意味らしいっす。

| Layer | カーネルサイズ | フィルタ数 | ストライド| パディング | 活性化関数 | 
|:---:|:---:|:---:|:---:|:---:|:---:|
| Input | 224 x 224 x 3 (入力サイズ) |
| Convolution | 3 x 3 | 64 | 1 | 1 | ReLU |
| Convolution | 3 x 3 | 64 | 1 | 1 | ReLU |
| MaxPooling | 2 x 2 | 2 | 0 | - | - |
| Convolution | 3 x 3 | 128 | 1 | 1 | ReLU |
| Convolution | 3 x 3 | 128 | 1 | 1 | ReLU |
| MaxPooling | 2 x 2 | 2 | 0 | - | - |
| Convolution | 3 x 3 | 256 | 1 | 1 | ReLU |
| Convolution | 3 x 3 | 256 | 1 | 1 | ReLU |
| Convolution | 3 x 3 | 256 | 1 | 1 | ReLU |
| MaxPooling | 2 x 2 | 2 | 0 | - | - |
| Convolution | 3 x 3 | 512 | 1 | 1 | ReLU |
| Convolution | 3 x 3 | 512 | 1 | 1 | ReLU |
| Convolution | 3 x 3 | 512 | 1 | 1 | ReLU |
| MaxPooling | 2 x 2 | 2 | 0 | - | - |
| Convolution | 3 x 3 | 512 | 1 | 1 | ReLU |
| Convolution | 3 x 3 | 512 | 1 | 1 | ReLU |
| Convolution | 3 x 3 | 512 | 1 | 1 | ReLU |
| MaxPooling | 2 x 2 | 2 | 0 | - | - |
| MultiLayerPerceptron | 4096 | - | - | - | - | ReLU + Dropout |
| MultiLayerPerceptron | 4096 | - | - | - | - | ReLU + Dropout |
| MultiLayerPerceptron | 2 (クラス) | - | - | - | - | Softmax|

答え
- Pytorch [scripts_pytorch/vgg16_pytorch.py](scripts_pytorch/vgg16_pytorch.py)
- Tensorflow [scripts_tf_layers/VGG16_tf_layers.py](scripts_tf_layers/VGG16_tf_layers.py)
- Keras [scripts_keras/vgg16_keras.py](scripts_keras/vgg16_keras.py)
- chainer [scripts_chainer/vgg16_chainer.py](scripts_chainer/vgg16_chainer.py)

## Q. VGG19

元論文 >> Very Deep Convolutional Networks for Large-Scale Image Recognition https://arxiv.org/abs/1409.1556 (2014)

VGG19はVGG16にConvolutionが3つ増えたバージョン。こっちよりもVGG16のほうがよく使われてます。多分認識精度とパラメータ数が割に合わないのかと。とりあえずモデル理解のために実装してみましょう。

| Layer | カーネルサイズ | フィルタ数 | ストライド| パディング | 活性化関数 | 
|:---:|:---:|:---:|:---:|:---:|:---:|
| Input | 224 x 224 x 3 (入力サイズ) |
| Convolution | 3 x 3 | 64 | 1 | 1 | ReLU |
| Convolution | 3 x 3 | 64 | 1 | 1 | ReLU |
| MaxPooling | 2 x 2 | 2 | 0 | - | - |
| Convolution | 3 x 3 | 128 | 1 | 1 | ReLU |
| Convolution | 3 x 3 | 128 | 1 | 1 | ReLU |
| MaxPooling | 2 x 2 | 2 | 0 | - | - |
| Convolution | 3 x 3 | 256 | 1 | 1 | ReLU |
| Convolution | 3 x 3 | 256 | 1 | 1 | ReLU |
| Convolution | 3 x 3 | 256 | 1 | 1 | ReLU |
| **Convolution** | 3 x 3 | 256 | 1 | 1 | ReLU |
| MaxPooling | 2 x 2 | 2 | 0 | - | - |
| Convolution | 3 x 3 | 512 | 1 | 1 | ReLU |
| Convolution | 3 x 3 | 512 | 1 | 1 | ReLU |
| Convolution | 3 x 3 | 512 | 1 | 1 | ReLU |
| **Convolution** | 3 x 3 | 512 | 1 | 1 | ReLU |
| MaxPooling | 2 x 2 | 2 | 0 | - | - |
| Convolution | 3 x 3 | 512 | 1 | 1 | ReLU |
| Convolution | 3 x 3 | 512 | 1 | 1 | ReLU |
| Convolution | 3 x 3 | 512 | 1 | 1 | ReLU |
| **Convolution** | 3 x 3 | 512 | 1 | 1 | ReLU |
| MaxPooling | 2 x 2 | 2 | 0 | - | - |
| MultiLayerPerceptron | 4096 | - | - | - | - | ReLU + Dropout |
| MultiLayerPerceptron | 4096 | - | - | - | - | ReLU + Dropout |
| MultiLayerPerceptron | 2 (クラス) | - | - | - | - | Softmax|

答え
- Pytorch [scripts_pytorch/vgg19_pytorch.py](scripts_pytorch/vgg19_pytorch.py)
- Tensorflow [scripts_tf_layers/VGG19_tf_layers.py](scripts_tf_layers/VGG19_tf_layers.py)
- Keras [scripts_keras/vgg19_keras.py](scripts_keras/vgg19_keras.py)
- chainer [scripts_chainer/vgg19_chainer.py](scripts_chainer/vgg19_chainer.py)


## Q. GoogLeNet-v1

元論文 >> Going Deeper with Convolutions https://arxiv.org/abs/1409.4842 (2014)

- Pytorch [scripts_pytorch/googlenetv1_pytorch.py](scripts_pytorch/googlenetv1_pytorch.py)
- Tensorflow [answers/googlenetv1_tensorflow_slim.py](answers/googlenetv1_tensorflow_slim.py)
- Keras [scripts_keras/googlenetv1_keras.py](scripts_keras/googlenetv1_keras.py)
- chainer [scripts_chainer/bn_chainer.py](scripts_chainer/bn_chainer.py)


##  Q. Batch Normalization

元論文 >> Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift https://arxiv.org/abs/1502.03167 (2015)

Batch normalizationとは学習をめちゃくちゃ効率化するための手法です。今ではどのディープラーニングでもBNはかかせない存在となっています。

論文中では学習がうまくいかないのは、そもそもconvolutionしまくると値がシフトしてしまうことが原因だといってます。これが共分散シフトです。なのでBNはlayerの出力に対して正規化を行うことでこのシフトをなくすのが目的となっています。

ここではVGG16のconvの後にBNを実装してみましょう。

pytorchでは*torch.nn.BatchNorm2d()*, tensorflowでは*tf.layers.batch_normalization(), tf.layers.BatchNormalization()*, Kerasでは*keras.layers.BatchNormalization()* ,chainerでは*chainer.links.BatchNormalization()* で実装できる。

答え
- Pytorch [scripts_pytorch/bn_pytorch.py](scripts_pytorch/bn_pytorch.py)
- Tensorflow [scripts_tf_layers/BN_tf_layers.py](scripts_tf_layers/BN_tf_layers.py)
- Keras [scripts_keras/bn_keras.py](scripts_keras/bn_keras.py)
- chainer [scripts_chainer/bn_chainer.py](scripts_chainer/bn_chainer.py)

## Q. ResNet

元論文 >> Deep Residual Learning for Image Recognition https://arxiv.org/abs/1512.03385 (2015)

Skip connectionによって、Lossのback propagationの効率化を行った。

<img src="assets/resnet_block.jpg" width=400>


答え

#### Res18

- Pytorch [scripts_pytorch/res18_pytorch.py](scripts_pytorch/res18_pytorch.py)
- TensorFlow [scripts_tf_layers/Res18_tf_layers.py](scripts_tf_layers/Res18_tf_layers.py)
- Keras [scripts_keras/res18_keras.py](scripts_keras/res18_keras.py)

#### Res34

- Pytorch [scripts_pytorch/res34_pytorch.py](scripts_pytorch/res34_pytorch.py)
- TensorFlow [scripts_tf_layers/Res34_tf_layers.py](scripts_tf_layers/Res34_tf_layers.py)
- Keras [scripts_keras/res34_keras.py](scripts_keras/res34_keras.py)

#### Res50

- Pytorch [scripts_pytorch/res50_pytorch.py](scripts_pytorch/res50_pytorch.py)
- TensorFlow [scripts_tf_layers/Res50_tf_layers.py](scripts_tf_layers/res50_tf_layers.py)
- Tensorflow[answers/res50_tensorflow_slim.py](answers/res50_tensorflow_slim.py)
- Keras [scripts_keras/res50_keras.py](scripts_keras/res50_keras.py)
- chainer(WiP) [scripts_chainer/res50_chainer.py](scripts_chainer/res50_chainer.py)

#### Res101

- Pytorch [scripts_pytorch/res101_pytorch.py](scripts_pytorch/res101_pytorch.py)
- TensorFlow [scripts_tf_layers/Res101_tf_layers.py](scripts_pytorch/Res101_tf_layers.py)
- Keras [scripts_keras/res101_keras.py](scripts_keras/res101_keras.py)

#### Res152

- Pytorch [scripts_pytorch/res152_pytorch.py](scripts_pytorch/res152_pytorch.py)
- TensorFlow [scripts_tf_layers/Res152_tf_layers.py](scripts_tf_layers/Res152_tf_layers.py)
- Keras [scripts_keras/res152_keras.py](scripts_keras/res152_keras.py)

## Q. ResNeXt

元論文 >> Aggregated Residual Transformations for Deep Neural Networks https://arxiv.org/abs/1611.05431 (2016)

ResNetのblockを内を細かく分けて、それらの和をとることでネットワーク内にネットワークを複数作成するモジュールを実現している。
パラメータ数を減らしながらも精度向上を実現した。

![](assets/resnext_block.jpg)

#### ResNeXt50

- Pytorch [scripts_pytorch/resNeXt50_pytorch.py](scripts_pytorch/resNeXt50_pytorch.py)
- Tensorflow [scripts_tf_layers/ResNeXt50_tf_layers.py](scripts_tf_layers/ResNeXt50_tf_layers.py)
- Keras [scripts_keras/resNeXt50_keras.py](scripts_keras/resNeXt50_keras.py)

#### ResNeXt101

- Pytorch [scripts_pytorch/resNeXt101_pytorch.py](scripts_pytorch/resNeXt101_pytorch.py)
- Keras [scripts_tf_keras/resNeXt101_keras.py](scripts_keras/resNeXt101_keras.py)


## Q. Xception

元論文 >> Xception: Deep Learning with Depthwise Separable Convolutions https://arxiv.org/abs/1610.02357 (2016)

- Pytorch [scripts_pytorch/xception_pytorch.py](scripts_pytorch/xception_pytorch.py)
- TensorFlow [scripts_tf_layers/xception_tf_layers.py](scripts_tf_layers/Xception_tf_layers.py)
- Keras [scripts_keras/xception_keras.py](scripts_keras/xception_keras.py)

## Q. DenseNet

元論文 >> Densely Connected Convolutional Networks https://arxiv.org/abs/1608.06993 (2016)

ResNetのshortcut connectionは勾配を直接的に前のLayerに伝えられるけども、shorcut connectionをLayerを通した出力に足して(sum)いるのは情報のフローを妨げる可能性があるらしい。

そこでconcatをするのだが、DenseBLockというconvolutionの塊の全てのLayerをconcatにする。

#### DenseNet121

- Pytorch [scripts_pytorch/DenseNet121_pytorch.py](scripts_pytorch/DenseNet121_pytorch.py)

#### DenseNet169

- Pytorch [scripts_pytorch/DenseNet169_pytorch.py](scripts_pytorch/DenseNet169_pytorch.py)

#### DenseNet201

- Pytorch [scripts_pytorch/DenseNet201_pytorch.py](scripts_pytorch/DenseNet201_pytorch.py)

#### DenseNet264

- Pytorch [scripts_pytorch/DenseNet264_pytorch.py](scripts_pytorch/DenseNet264_pytorch.py)

## Q. MobileNet-v1

元論文 >> MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications https://arxiv.org/abs/1704.04861?source=post_page--------------------------- (2017)

普通のConvolutionは、カーネルサイズ Dk x Dkを持っていて、出力のチャネルNだけ用意される。
これに入力のチャネルM、かつ入力の特徴マップDf x Dfに対して適用されるので、全計算量は Dk Dk M N Df Df となる。

MobileNetではConvolutionを二つのconvolution操作(Depth-wise convとPoint-wise conv)に分解している。

<img src="assets/mobilenet-v1.png" width="400">

この分解により以下のようなConvolutionの操作になる。

| Standard Conv | Separable Conv |
|:---:|:---:|
| <img src="assets/standard_conv.png" width=350> | <img src="assets/separable_conv.png" width=400> |

Depth-wise convでは入力の特徴マップの各チャネルにdepth=1のカーネル（Dk x Dk)を適用する。Point-wise convでは(1 x 1 x M)のconvolutionを適用してチャネル方向の線形和を計算する。

これにより得られる計算量は、

- Depth-wise Convは Dk Dk M Df Df
- Point-wise Convは M N Df Df

なので、合計して、Dk Dk M Df Df + M N Df Df = M Df Df (Dk Dk + N)

普通のConvとの比を取ると、

M Df Df (Dk Dk + N) / Dk Dk M N Df Df = (Dk Dk + N) / Dk Dk N = 1 / N + 1 / Dk^2

となる。普通はConvのカーネルサイズはDk=3となることが多いので、MobileNetでは計算量を 1 / 9 に減らすことができる。

この仕組みでMobileNetではモデルの性能（精度）の損失をほとんど行わずに、パラメータ削減に成功している。

- Pytorch [scripts_pytorch/MobileNet_v1_pytorch.py](scripts_pytorch/MobileNet_v1_pytorch.py)
- TensorFlow [scripts_tf_layers/MobileNet_v1_tf_layers.py](scripts_tf_layers/MobileNet_v1_tf_layers.py)

## Q. MobileNet-v2

元論文 >> MobileNetV2: Inverted Residuals and Linear Bottlenecks https://arxiv.org/abs/1801.04381 (2018)

MobileNetを改良したMobileNet-v2では、Convolutionのやり方がさらに変更になっている。

v2ではこのようなconvolutionのブロックを重ねていく。
入力の特徴マップのチャネルをkとする。
tは拡大率で、論文ではt=6としている。

1. Conv(kernel:1x1, channel:tk) + BatchNormalization + ReLU6
2. Depth wise Conv(kernel:1x1, channel:t, stride:(1 or 2)) + BatchNormalization + ReLU6
3. Conv(kernel:1x1, channel:k') + BatchNormalization + ReLU6
4. もし、strideが1なら、skip connectionを加える


<img src="assets/mobilenet-v2.png" width=500>

ちなみにReLU6とは

y = min(max(0, x), 6)

となった活性化関数を表す。

本家実装は https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet

- Pytorch [scripts_pytorch/MobileNet_v2_pytorch.py](scripts_pytorch/MobileNet_v2_pytorch.py)

## EfficientNet

元論文 >> EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks https://arxiv.org/abs/1905.11946 (2019)


EfficientNetは主に３つのモジュールがある。

1. Stem

入力画像に対して、以下操作を行う。

- Conv(channel = 32, kernel_size = 3, padding = 1, stride = 2)
- BatchNormalization
- Swish activation

2. Block

ブロックはいくつかの大きく７種類ある。各ブロックは繰り返し数(repeat)が設定されている。
各ブロックのパラメータは以下のとおり。

| Block | Kernel_size | repeats | filters_in | filters_out | expand_ratio | stride | se_ratio |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Block 1 | 3 | 1 | 32 | 16 | 1 | 1 | 0.25 |
| Block 2 | 3 | 2 | 16 | 24 | 6 | 2 | 0.25 |
| Block 3 | 5 | 2 | 24 | 40 | 6 | 2 | 0.25 |
| Block 4 | 3 | 3 | 40 | 80 | 6 | 2 | 0.25 |
| Block 5 | 5 | 3 | 80 | 112 | 6 | 1 | 0.25 |
| Block 6 | 5 | 4 | 112 | 192 | 6 | 2 | 0.25 |
| Block 7 | 3 | 1 | 192 | 320 | 6 | 1 | 0.25 |

ブロックをそれぞれのrepeatの数だけ繰り返す。

<img src="assets/EfficientNet_block.png" width=300>


3. Output

最後は以下の操作をして、クラス予測を出力する。
- Conv(channel = 1280, kernel_size = 1, padding = 0)
- BatchNormalization
- Swish activation
- Global Average Pooling
- BatchNormalization
- Dropout
- Fully Connected (class number)
- Softmax

以下の実装は https://github.com/keras-team/keras-applications/blob/master/keras_applications/efficientnet.py を参照した

### EfficientNetB0

- Pytorch [scripts_pytorch/EfficientNetB0_pytorch.py](scripts_pytorch/EfficientNetB0_pytorch.py)

### EfficientNetB1

- Pytorch [scripts_pytorch/EfficientNetB1_pytorch.py](scripts_pytorch/EfficientNetB1_pytorch.py)

### EfficientNetB2

- Pytorch [scripts_pytorch/EfficientNetB2_pytorch.py](scripts_pytorch/EfficientNetB2_pytorch.py)

### EfficientNetB3

- Pytorch [scripts_pytorch/EfficientNetB3_pytorch.py](scripts_pytorch/EfficientNetB3_pytorch.py)

### EfficientNetB4

- Pytorch [scripts_pytorch/EfficientNetB4_pytorch.py](scripts_pytorch/EfficientNetB4_pytorch.py)

### EfficientNetB5

- Pytorch [scripts_pytorch/EfficientNetB5_pytorch.py](scripts_pytorch/EfficientNetB5_pytorch.py)

### EfficientNetB6

- Pytorch [scripts_pytorch/EfficientNetB6_pytorch.py](scripts_pytorch/EfficientNetB6_pytorch.py)

### EfficientNetB7

- Pytorch [scripts_pytorch/EfficientNetB7_pytorch.py](scripts_pytorch/EfficientNetB7_pytorch.py)