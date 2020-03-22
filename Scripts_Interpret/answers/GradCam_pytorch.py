import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
from glob import glob
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import OrderedDict

Class_label = ['akahara', 'madara']
Class_N = len(Class_label)
img_height, img_width = 128, 128
channel = 3

# GPU
GPU = False
device = torch.device("cuda" if GPU and torch.cuda.is_available() else "cpu")

# random seed
torch.manual_seed(0)


class ResBlock(torch.nn.Module):
    def __init__(self, in_f, f_1, out_f, stride=1):
        super(ResBlock, self).__init__()

        self.stride = stride
        self.fit_dim = False

        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_f, f_1, kernel_size=1, padding=0, stride=stride),
            torch.nn.BatchNorm2d(f_1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(f_1, f_1, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(f_1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(f_1, out_f, kernel_size=1, padding=0, stride=1),
            torch.nn.BatchNorm2d(out_f),
            torch.nn.ReLU()
        )

        if in_f != out_f:
            self.fit_conv = torch.nn.Conv2d(in_f, out_f, kernel_size=1, padding=0, stride=1)
            self.fit_bn = torch.nn.BatchNorm2d(out_f)
            self.fit_dim = True
            
            
        
    def forward(self, x):
        res_x = self.block(x)
        
        if self.fit_dim:
            x = self.fit_conv(x)
            x = self.fit_bn(x)
            x = F.relu(x)
        
        if self.stride == 2:
            x = F.max_pool2d(x, 2, stride=2)
            
        x = torch.add(res_x, x)
        x = F.relu(x)
        return x

        
class Res101(torch.nn.Module):
    def __init__(self):
        super(Res101, self).__init__()

        self.conv1 = torch.nn.Conv2d(channel, 64, kernel_size=7, padding=3, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(64)
        
        self.resblock2_1 = ResBlock(64, 64, 256)
        self.resblock2_2 = ResBlock(256, 64, 256)
        self.resblock2_3 = ResBlock(256, 64, 256)

        self.resblock3_1 = ResBlock(256, 128, 512, stride=2)
        self.resblock3_2 = ResBlock(512, 128, 512)
        self.resblock3_3 = ResBlock(512, 128, 512)
        self.resblock3_4 = ResBlock(512, 128, 512)

        self.resblock4_1 = ResBlock(512, 256, 1024, stride=2)
        block = []
        for _ in range(22):
            block.append(ResBlock(1024, 256, 1024))
        self.resblock4s = torch.nn.Sequential(*block)

        self.resblock5_1 = ResBlock(1024, 512, 2048, stride=2)
        self.resblock5_2 = ResBlock(2048, 512, 2048)
        self.resblock5_3 = ResBlock(2048, 512, 2048)
        
        self.linear = torch.nn.Linear(2048, Class_N)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3, padding=1, stride=2)

        x = self.resblock2_1(x)
        x = self.resblock2_2(x)
        x = self.resblock2_3(x)

        x = self.resblock3_1(x)
        x = self.resblock3_2(x)
        x = self.resblock3_3(x)
        x = self.resblock3_4(x)

        x = self.resblock4_1(x)
        x = self.resblock4s(x)

        x = self.resblock5_1(x)
        x = self.resblock5_2(x)
        x = self.resblock5_3(x)

        x = F.avg_pool2d(x, [img_height//32, img_width//32], padding=0, stride=1)
        x = x.view(list(x.size())[0], -1)
        x = self.linear(x)
        x = F.softmax(x, dim=1)
        
        return x



# get train data
def data_load(path, hf=False, vf=False, rot=False):
    xs = []
    ts = []
    paths = []
    
    for dir_path in glob(path + '/*'):
        for path in glob(dir_path + '/*'):
            x = cv2.imread(path)
            x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
            x /= 255.
            x = x[..., ::-1]
            xs.append(x)

            for i, _Class_label in enumerate(Class_label):
                if _Class_label in path:
                    t = i
            
            ts.append(t)

            paths.append(path)

            if hf:
                xs.append(x[:, ::-1])
                ts.append(t)
                paths.append(path)

            if vf:
                xs.append(x[::-1])
                ts.append(t)
                paths.append(path)

            if hf and vf:
                xs.append(x[::-1, ::-1])
                ts.append(t)
                paths.append(path)

            if rot != False:
                angle = rot
                scale = 1

                # show
                a_num = 360 // rot
                w_num = np.ceil(np.sqrt(a_num))
                h_num = np.ceil(a_num / w_num)
                count = 1
                #plt.subplot(h_num, w_num, count)
                #plt.axis('off')
                #plt.imshow(x)
                #plt.title("angle=0")
                
                while angle < 360:
                    _h, _w, _c = x.shape
                    max_side = max(_h, _w)
                    tmp = np.zeros((max_side, max_side, _c))
                    tx = int((max_side - _w) / 2)
                    ty = int((max_side - _h) / 2)
                    tmp[ty: ty+_h, tx: tx+_w] = x.copy()
                    M = cv2.getRotationMatrix2D((max_side/2, max_side/2), angle, scale)
                    _x = cv2.warpAffine(tmp, M, (max_side, max_side))
                    _x = _x[tx:tx+_w, ty:ty+_h]
                    xs.append(_x)
                    ts.append(t)
                    paths.append(path)

                    # show
                    #count += 1
                    #plt.subplot(h_num, w_num, count)
                    #plt.imshow(_x)
                    #plt.axis('off')
                    #plt.title("angle={}".format(angle))

                    angle += rot
                #plt.show()


    xs = np.array(xs, dtype=np.float32)
    ts = np.array(ts, dtype=np.int)
    
    xs = xs.transpose(0,3,1,2)

    return xs, ts, paths



# train
def train():
    # model
    model = Res101().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()

    xs, ts, paths = data_load('../Dataset/train/images/', hf=True, vf=True, rot=10)

    # training
    mb = 32
    mbi = 0
    train_ind = np.arange(len(xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)

    loss_fn = torch.nn.NLLLoss()
    
    for i in range(500):
        if mbi + mb > len(xs):
            mb_ind = copy.copy(train_ind)[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
        else:
            mb_ind = train_ind[mbi: mbi+mb]
            mbi += mb

        x = torch.tensor(xs[mb_ind], dtype=torch.float).to(device)
        t = torch.tensor(ts[mb_ind], dtype=torch.long).to(device)

        opt.zero_grad()
        y = model(x)
        #y = F.log_softmax(y, dim=1)
        loss = loss_fn(torch.log(y), t)
        
        loss.backward()
        opt.step()
    
        pred = y.argmax(dim=1, keepdim=True)
        acc = pred.eq(t.view_as(pred)).sum().item() / mb

        if (i + 1) % 50 == 0:
            print("iter >>", i+1, ', loss >>', loss.item(), ', accuracy >>', acc)

    torch.save(model.state_dict(), 'Res101.pt')

# test
def test(target_layer_name):
    model = Res101().to(device)
    model.eval()
    model.load_state_dict(torch.load('res101.pt', map_location=torch.device(device)))

    xs, ts, paths = data_load('../Dataset/test/images/')

    target_layer = None

    for name, module in model.named_modules():
      if target_layer_name == name:
        print('target:', name)
        target_layer = module

    if target_layer is None:
      for name, module in model.named_modules():
        print(name)
      raise Exception('invalid target layer name >>', target_layer_name)

    if type(target_layer) is torch.nn.Sequential:
      target_layer = target_layer[-1]

    print(target_layer)

    fmap_pool = OrderedDict()
    grad_pool = OrderedDict()

    def forward_hook(key):
        def forward_hook_(module, input, output):
            # Save featuremaps
            fmap_pool[key] = output.detach()

        return forward_hook_

    def backward_hook(key):
        def backward_hook_(module, grad_in, grad_out):
            # Save the gradients correspond to the featuremaps
            grad_pool[key] = grad_out[0].detach()

        return backward_hook_

    # If any candidates are not specified, the hook is registered to all the layers.
    for name, module in model.named_modules():
            module.register_forward_hook(forward_hook(name))
            module.register_backward_hook(backward_hook(name))


    for i in range(len(paths)):
        _x = xs[i]
        t = ts[i]
        path = paths[i]
        
        x = np.expand_dims(_x, axis=0)
        x = torch.tensor(x, dtype=torch.float).to(device)
        
        # forward network
        logit = model(x)
        pred = F.softmax(logit, dim=1).detach().cpu().numpy()

        raw_image = (_x ).transpose(1, 2, 0)

        plt.subplot(1, Class_N + 1, 1)
        plt.imshow(raw_image)
        plt.title('input')

        for i, class_label in enumerate(Class_label):
            # set one-hot class activity
            class_index = torch.zeros(pred.shape).to(device)

            _index = Class_label.index(class_label)
            class_index[:, _index] = 1

            logit.backward(gradient=class_index, retain_graph=True)
            
            #target_layer_output = target_layer.forward(x)
            fmaps = fmap_pool[target_layer_name]
            grads = grad_pool[target_layer_name]
            weights = F.adaptive_avg_pool2d(grads, 1)

            gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
            gcam = F.relu(gcam)

            gcam = F.interpolate(gcam, [img_height, img_width], mode="bilinear", align_corners=False)

            B, C, H, W = gcam.shape
            gcam = gcam.view(B, -1)
            gcam -= gcam.min(dim=1, keepdim=True)[0]
            gcam /= gcam.max(dim=1, keepdim=True)[0]
            gcam = gcam.view(B, C, H, W)

            gcam = gcam.cpu().numpy()[0, 0]
            cmap = cm.jet_r(gcam)[..., :3]
            gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
        
            plt.subplot(1, Class_N + 1, i + 2)
            plt.imshow(gcam)
            plt.title('{} :{:.2f}'.format(class_label, pred[0, i]))

        plt.show()

        print("in {}, predicted probabilities >> {}".format(path, pred))



def arg_parse():
    parser = argparse.ArgumentParser(description='CNN implemented with Keras')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--target', dest='target_layer', default='conv3', type=str)
    args = parser.parse_args()
    return args

# main
if __name__ == '__main__':
    args = arg_parse()

    if args.train:
        train()
    if args.test:
        test(args.target_layer)

    if not (args.train or args.test):
        print("please select train or test flag")
        print("train: python main.py --train")
        print("test:  python main.py --test")
        print("both:  python main.py --train --test")
