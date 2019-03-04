# Caffeインストール(Docker環境)

ネイティブ環境...OSに直にって意味。
ここではUbuntu16.04を想定。Docker時のインストール方法。
docker使用時ではコマンドを使う時にDocker上かdockerをはなれるかが違う。Caffeをgitからダウンロードする以外はdocker上で行う。

## パッケージのインストール

下記のコピペで必要なパッケージをインストールする。
**docker上で**

```bash
$ apt update && apt upgrade

$ apt install -y python python-pip python3-pip python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose python-tk python-yaml

$ apt install -y build-essential cmake git pkg-config libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libatlas-base-dev libgflags-dev libgoogle-glog-dev liblmdb-dev libhdf5-dev wget emacs vim gedit sudo

$ apt install -y --no-install-recommends libboost-all-dev

$ apt install emacs vim

$ echo "alias pip='python -m pip'" >> ~/.bashrc

$ pip install opencv-python scikit-image protobuf easydict cython
```

## CUDAのダウンロード

**docker上で**

これはpy-faster-rcnnは必要で、普通のcaffeは必要ないと思いますが、一応。
CUDAをダウンロードする必要があります。
CUDAの配布サイトからubuntu用cudaを取って、ホームのどっかにおく。
　Linux -> x86_64 -> Ubuntu -> 16.04 -> deb(network)
　の順番に選択。あとはサイト上に指示が出るので、その通りにやってください。
　
　最終的に /usr/local/にcudaのディレクトリができると思うので、次に~/.bashrcを開いて、一番下に次のコードを追加してください！（CUDAのパスを指定します）

```bash
export CUDAHOME=/usr/local/cuda
```

## CUDNNのダウンロード

**docker上で**

cudnnの導入方法を調べたのでまとめます。
https://developer.nvidia.com/rdp/cudnn-download のサイトのDownload cuDNN v5.1 (Jan 20, 2017), for CUDA 8.0　から cuDNN v5.1 Library for Linux をダウンロードする。(CUDAのバージョンをちゃんと合わせて！！)
-- あまりバージョンが新しいとtensorfowとかのバージョンが合わないとか出そうなのであえて古いバージョンで
-- py-faster-rcnnのときはcudnn-v4.0じゃないとmakeできないので、Download cuDNN v4 (Feb 10, 2016), for CUDA 7.0 and later　から　cuDNN v4 Library for Linux (updated October 18th,2016) をダウンロードして下さい
このtarファイルを下記コマンドで展開する。

```bash
$ tar xzvf cudnn-7.0-linux-x64-v4.0-prod.tgz
```

次に下記コマンドでcudnnのファイルをcudaにコピーするといいらしいです。

```bash
$ sudo cp -a cuda/lib64/* /usr/local/cuda-8.0/lib64/
$ sudo cp -a cuda/include/* /usr/local/cuda-8.0/include
```

最後にldconfigをする。

```bash
$ sudo ldconfig
```

## Caffeのダウンロード

Gitからcaffe一式を下記コマンドで取ってくる。
**dockerを離れて**

```bash
$ git clone --recursive https://github.com/BVLC/caffe.git
```

## Caffeのコンパイル

**docker上で**

caffeのディレクトリに移動する。

```bash
$ cd caffe
```

次に下記コマンドでMakefile.configを作成する。

```bash
cp Makefile.config.example Makefile.config
```

Makefile.configの中身を次のように変更する。

- CUDNNを使う時は CUDNN:= 1　をアンコメント
- CPUのみで使う時は CPU_ONLY:= 1　をアンコメント
- OpenCV-3.で使う時は OPENCV_VERSION:= 3 をアンコメント
- WITH_PYTHON_LAYER:= 1をアンコメント
- INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include 　/usr/include/hdf5/serial に書き直す
- LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib　/usr/lib/x86_64-linux-gnu/hdf5/serial　に書き直す
- CUDA_ARCHを次のように変更する

```bash
CUDA_ARCH := -gencode arch=compute_30,code=sm_30 \
    -gencode arch=compute_35,code=sm_35 \
    -gencode arch=compute_50,code=sm_50 \
    -gencode arch=compute_50,code=compute_50
```
- dockerではnumpyのパスがネイティブとは違うので、

```bash
PYTHON_INCLUDE := /usr/include/python2.7 \
    /usr/local/lib/python2.7/dist-packages/numpy/core/include
```
に書き換える。

終わったらmakeする。

```bash
$ make all -j8
```

次にpython用のmakeをする。

```bash
$ make pycaffe
```

こうなればインストールはOK。

```bash
caffe$ make pycaffe
CXX/LD -o python/caffe/_caffe.so python/caffe/_caffe.cpp
touch python/caffe/proto/__init__.py
PROTOC (python) src/caffe/proto/caffe.proto
caffe$ 
```

最後に~/.bashrcの最後に次のコードを追記する。

```bash
export PYTHONPATH=~/caffe/python:$PYTHONPATH
```

最後に、ホームに戻って.bashrcの変更を反映させる。

```bash
$ cd ~
$ source .bashrc
```

あとはpythonを開いてimport caffeできればOK

```bash
$ python
>> import caffe
>>
```

## エラー集

###  /usr/bin/ld: cannot find -lcblas

 apt install -y libatlas-base-dev で解決

### Cannot find hdf5.so
1. apt install -y libhdf5-dev
2. 次に　cd /usr/lib/x86_64-linux-gnu
3. この中にlibhdf5_serial.so.#.#.# と libhdf5_serial_hl.so.#.#.# ファイルがあるはずなので、　それぞれに次のコマンドでリンクを貼る
4. sudo ln -s libhdf5_serial.so.#.#.# libhdf5.so
5. sudo ln -s libhdf5_serial_hl.so.#.#.# libhdf5_hl.so


### undefined reference cv:imencode()

古いcaffeだとopencvのバージョンが3を指定できない（OPENCV-VERSION:=3　がない）。それでこのエラーが出る。

これは Makefile のLIBRALIESに opencv_imgcodecs を追加すると多分makeが通るようになるはずです。

### cannot find cublas_v2.h

```bash
src/caffe/parallel.cpp:2:26: fatal error: cuda_runtime.h: そのようなファイルやディレクトリはありません
#include 
^
compilation terminated.
make: *** [.build_release/src/caffe/parallel.o] エラー 1
make: *** 未完了のジョブを待っています....
In file included from ./include/caffe/common.hpp:19:0,
from ./include/caffe/blob.hpp:8,
from src/caffe/blob.cpp:4:
./include/caffe/util/device_alternate.hpp:34:23: fatal error: cublas_v2.h: そのようなファイルやディレクトリはありません
#include
```
というエラーが出たら、Makefile.configの次の部分を変更する。

```bash
BLAS_INCLUDE := /usr/local/cuda/targets/x86_64-linux/include
BLAS_LIB := /usr/local/cuda/targets/x86_64-linux/lib
```

### make pycaffe時のエラー

```bash
CXX/LD -o python/caffe/caffe.so python/caffe/_caffe.cpp
python/caffe/_caffe.cpp:11:31: fatal error: numpy/arrayobject.h: No such file or directory compilation terminated.
Makefile:489: recipe for target 'python/caffe/caffe.so' failed
make: *** [python/caffe/_caffe.so] Error 1
```
このエラーはMakefile.configで指定したnumpyのパスが違う時に起きる。
pythonを起動して次のコマンドでnumpyの位置を調べる。

```bash
:~ :$ python
Python 2.7.13 (default, Apr  4 2017, 08:47:57) 
[GCC 4.2.1 Compatible Apple LLVM 8.1.0 (clang-802.0.38)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> 
>>> import numpy
>>> numpy.__file__
'/usr/local/lib/python2.7/site-packages/numpy/__init__.pyc'
>>>
```

これがnumpyのパス。Makefile.config の PYTHON_INCLUDE := /usr/include/python2.7 \ /usr/lib/python2.7/dist-packages/numpy/core/include を上のパスに書き換える。
