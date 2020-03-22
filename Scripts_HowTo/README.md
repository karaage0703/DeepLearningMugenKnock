# Q. 3. CNNでの各フレームワークの使い方

## フレームワーク早見表

| | PyTorch | Tensorflow | Keras | Chainer | Caffe |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 入力 | [mb,c,h,w] | [mb, h, w, c] | [mb, h, w, c] | [mc, c, h, w] | [mb, c, h, w] |
| 教師ラベル | index [mb] | onehot [mb, cls] | onehot [mb, cls] | index [mb] | index [mb] |
| 速度 | まあまあ早い | 早い | 早い | 普通 | まあまあ早い？ |


## Q.1. 共通事項

ディープラーニングのフレームワークは今たくさんある。
よく聞く名前だと、PyTorhc, Chainer, Tensorflow, Keras, Caffeとか、あとはdarknetなんてのもある。
それぞれの特徴は

| | |
|:---:|:---:|
| PyTorch | Face◯ook社が開発してる。最近キテる系 。使いやすい。いろんな手法を実装しているので参考にしやすい |
| Tensorflow | Goo◯le社が開発している。sessionという独特な使い回しがある。慣れればどうってことない。|
| Keras | TensorflowとかTheanoをベースにして使いやすくしている。殆ど高レベルAPI化していてコードが短クて済む反面、コアな部分がいじりづらい。コアな研究をするときは扱いにくいと思う。|
| Chainer | 日本のPrefer◯ed Networks社が開発してる。日本語の説明が多い。pytorchとかなり似ている。ただちょくちょく使い方が分かりにくいものがある|
| Caffe | 一つの研究室から開発が始まった。ディープラーニングのフレームの老舗的存在。上のと違ってコンパイルする必要があり、そこがかなり厄介。モデルの定義もしにくい反面、実行速度は早く、モデルも思ったより柔軟に書ける。|


コードを書く時はどれも共通している流れがあり、
1. 学習データの用意
2. モデルの定義 VGG? ResNet?
3. モデルの最適化手法を設定 SGD? Adam?
4. 学習する

この流れさえ掴めればフレームワークが違っても苦労しなくなる。

ただし、それぞれで**独特さ、アイデンティティ**があるので注意。

| | |
|:---:|:---:|
| PyTorch | torch.tensorという独自の型を使う。モデル定義が少しめんどい |
| Chainer | chainer.Variable という独自の型を使う。モデル定義が少しめんどい  |
| Tensorflow | sessionという独特な仕組みを使う。モデルを定義→sessionに入れて計算みたいな流れを取る。 |
| Keras | 上に比べて結構シンプルに書ける。独特な型とかを意識せずに使える。ただしコアな改造がしにくい。 |
| Caffe | モデル定義を別ファイルに用意しなきゃいけないのでけっこうめんどう。コードよりもファイルの数が必然的に多くなる。　それ以外はシンプルに書けて意外と便利な部分が多い。|

CPUでの計算時間の速さでいうと、体感的に **Tensorflow, Keras(Tensorflow) >>> PyTorch, Caffe > Chainer** という感じがする。さすがgo◯gle


## 2. PyTorchの使い方

[README_pytorch.md](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_howto/README_pytorch.md)

## 3. Tensorflowの使い方

[README_tensorflow.md](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_howto/README_tensorflow.md)
## 4. Kerasの使い方

[README_keras.md](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_howto/README_keras.md)

## 5. Chainerの使い方

[README_chainer.md](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_howto/README_chaienr.md)

## 6. Caffeの使い方

Caffeはインストールのハードルが高いので注意。インストールではDockerでUbuntuを使うのがおすすめ。

インストール

- Docker環境 >> [README_caffe_install_docker.md](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_howto/README_caffe_install_docker.md)
- Native環境 >> [README_caffe_install_native.md](https://github.com/yoyoyo-yo/DeepLearningMugenKnock/blob/master/Question_howto/README_caffe_install_native.md)

