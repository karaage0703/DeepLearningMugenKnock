import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
from glob import glob

num_classes = 2
img_height, img_width = 224, 224
channel = 3
GPU = False
torch.manual_seed(0)


class InceptionModule(torch.nn.Module):
    def __init__(self, in_f, f_1, f_2_1, f_2_2, f_3_1, f_3_2, f_4_2):
        super(InceptionModule, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_f, f_1, kernel_size=1, padding=0, stride=1)
        
        self.conv2_1 = torch.nn.Conv2d(in_f, f_2_1, kernel_size=1, padding=0, stride=1)
        self.conv2_2 = torch.nn.Conv2d(f_2_1, f_2_2, kernel_size=3, padding=1, stride=1)
        
        self.conv3_1 = torch.nn.Conv2d(in_f, f_3_1, kernel_size=1, padding=0, stride=1)
        self.conv3_2 = torch.nn.Conv2d(f_3_1, f_3_2, kernel_size=5, padding=2, stride=1)

        self.conv4_2 = torch.nn.Conv2d(in_f, f_4_2, kernel_size=1, padding=0, stride=1)

        
    def forward(self, x):
        x1 = torch.nn.ReLU()(self.conv1(x))
        
        x2 = torch.nn.ReLU()(self.conv2_1(x))
        x2 = torch.nn.ReLU()(self.conv2_2(x2))

        x3 = torch.nn.ReLU()(self.conv3_1(x))
        x3 = torch.nn.ReLU()(self.conv3_2(x3))

        x4 = F.max_pool2d(x, 3, padding=1, stride=1)
        x4 = torch.nn.ReLU()(self.conv4_2(x4))
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x

        

class GoogLeNetv1(torch.nn.Module):
    def __init__(self):
        super(GoogLeNetv1, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, padding=0, stride=2)
        self.conv2_1 = torch.nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1)
        self.conv2_2 = torch.nn.Conv2d(64, 192, kernel_size=3, padding=1, stride=1)

        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)

        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.linear = torch.nn.Linear(1024, num_classes)
            
        self.aux1_conv1 = torch.nn.Conv2d(512, 128, kernel_size=1, padding=0, stride=1)
        self.aux1_linear1 = torch.nn.Linear(25088, 1024)
        self.aux1_linear2 = torch.nn.Linear(1024, num_classes)

        self.aux2_conv1 = torch.nn.Conv2d(528, 128, kernel_size=1, padding=0, stride=1)
        self.aux2_linear1 = torch.nn.Linear(25088, 1024)
        self.aux2_linear2 = torch.nn.Linear(1024, num_classes)

        
        
    def forward(self, x):
        x = torch.nn.ReLU()(self.conv1(x))
        x = F.max_pool2d(x, 3, padding=1, stride=2)
        x = torch.nn.modules.normalization.LocalResponseNorm(size=1)(x)

        x = torch.nn.ReLU()(self.conv2_1(x))
        x = torch.nn.ReLU()(self.conv2_2(x))
        x = torch.nn.modules.normalization.LocalResponseNorm(size=1)(x)
        x = F.max_pool2d(x, 3, padding=1, stride=2)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = F.max_pool2d(x, 3, padding=1, stride=2)

        x = self.inception4a(x)

        x_aux1 = F.avg_pool2d(x, 5, padding=2, stride=1)
        x_aux1 = torch.nn.ReLU()(self.aux1_conv1(x_aux1))
        x_aux1 = x_aux1.view(list(x_aux1.size())[0], -1)
        x_aux1 = torch.nn.ReLU()(self.aux1_linear1(x_aux1))
        x_aux1 = torch.nn.Dropout(p=0.7)(x_aux1)
        x_aux1 = self.aux1_linear2(x_aux1)
        x_aux1 = F.softmax(x_aux1, dim=1)
        
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        x_aux2 = F.avg_pool2d(x, 5, padding=2, stride=1)
        x_aux2 = torch.nn.ReLU()(self.aux2_conv1(x_aux2))
        x_aux2 = x_aux2.view(list(x_aux2.size())[0], -1)
        x_aux2 = torch.nn.ReLU()(self.aux2_linear1(x_aux2))
        x_aux2 = torch.nn.Dropout(p=0.7)(x_aux2)
        x_aux2 = self.aux2_linear2(x_aux2)
        x_aux2 = F.softmax(x_aux2, dim=1)

        x = self.inception4e(x)
        x = F.max_pool2d(x, 3, padding=1, stride=2)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = F.avg_pool2d(x, 7, padding=0, stride=1)
        x = x.view(list(x.size())[0], -1)
        x = self.linear(x)
        x = F.softmax(x, dim=1)
        
        return x, x_aux1, x_aux2


    
CLS = ['akahara', 'madara']

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

            for i, cls in enumerate(CLS):
                if cls in path:
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
    # GPU
    device = torch.device("cuda" if GPU else "cpu")

    # model
    model = GoogLeNetv1().to(device)
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
            mb_ind = train_ind[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
        else:
            mb_ind = train_ind[mbi: mbi+mb]
            mbi += mb

        x = torch.tensor(xs[mb_ind], dtype=torch.float).to(device)
        t = torch.tensor(ts[mb_ind], dtype=torch.long).to(device)

        opt.zero_grad()
        y, y_aux1, y_aux2 = model(x)
        #y = F.log_softmax(y, dim=1)
        loss = loss_fn(torch.log(y), t)
        loss_aux1 = loss_fn(torch.log(y_aux1), t)
        loss_aux2 = loss_fn(torch.log(y_aux2), t)

        loss = loss + loss_aux1 + loss_aux2
        
        loss.backward()
        opt.step()
    
        pred = y.argmax(dim=1, keepdim=True)
        acc = pred.eq(t.view_as(pred)).sum().item() / mb
        
        print("iter >>", i+1, ',loss >>', loss.item(), ',accuracy >>', acc)

    torch.save(model.state_dict(), 'cnn.pt')

# test
def test():
    device = torch.device("cuda" if GPU else "cpu")
    model = GoogLeNetv1().to(device)
    model.eval()
    model.load_state_dict(torch.load('cnn.pt'))

    xs, ts, paths = data_load('../Dataset/test/images/')

    with torch.no_grad():
        for i in range(len(paths)):
            x = xs[i]
            t = ts[i]
            path = paths[i]
            
            x = np.expand_dims(x, axis=0)
            x = torch.tensor(x, dtype=torch.float).to(device)
            
            pred = model(x)
            pred = pred.detach().cpu().numpy()[0]
        
            print("in {}, predicted probabilities >> {}".format(path, pred))
    

def arg_parse():
    parser = argparse.ArgumentParser(description='CNN implemented with Keras')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    args = parser.parse_args()
    return args

# main
if __name__ == '__main__':
    args = arg_parse()

    if args.train:
        train()
    if args.test:
        test()

    if not (args.train or args.test):
        print("please select train or test flag")
        print("train: python main.py --train")
        print("test:  python main.py --test")
        print("both:  python main.py --train --test")
