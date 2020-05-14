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

class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        conv1 = []
        for i in range(2):
            f = channel if i == 0 else 64
            conv1.append(torch.nn.Conv2d(f, 64, kernel_size=3, padding=1, stride=1))
            conv1.append(torch.nn.BatchNorm2d(64))
            conv1.append(torch.nn.ReLU())
        self.conv1 = torch.nn.Sequential(*conv1)
        
        conv2 = []
        for i in range(2):
            f = 64 if i == 0 else 128
            conv2.append(torch.nn.Conv2d(f, 128, kernel_size=3, padding=1, stride=1))
            conv2.append(torch.nn.BatchNorm2d(128))
            conv2.append(torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(*conv2)

        conv3 = []
        for i in range(3):
            f = 128 if i == 0 else 256
            conv3.append(torch.nn.Conv2d(f, 256, kernel_size=3, padding=1, stride=1))
            conv3.append(torch.nn.BatchNorm2d(256))
            conv3.append(torch.nn.ReLU())
        self.conv3 = torch.nn.Sequential(*conv3)
        
        conv4 = []
        for i in range(3):
            f = 256 if i == 0 else 512
            conv4.append(torch.nn.Conv2d(f, 512, kernel_size=3, padding=1, stride=1))
            conv4.append(torch.nn.BatchNorm2d(512))
            conv4.append(torch.nn.ReLU())
        self.conv4 = torch.nn.Sequential(*conv4)
            
        conv5 = []
        for i in range(3):
            conv5.append(torch.nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1))
            conv5.append(torch.nn.BatchNorm2d(512))
            conv5.append(torch.nn.ReLU())
        self.conv5 = torch.nn.Sequential(*conv5)
        
        
        self.fc1 = torch.nn.Linear(25088, 4096)
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.fc_out = torch.nn.Linear(4096, num_classes)
        
    def forward(self, x):
        # block conv1
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, stride=2, padding=0)

        # block conv2
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, stride=2, padding=0)

        # block conv3
        x = self.conv3(x)
        x = F.max_pool2d(x, 2, stride=2, padding=0)

        # block conv4
        x = self.conv4(x)
        x = F.max_pool2d(x, 2, stride=2, padding=0)

        # block conv5
        x = self.conv5(x)
        x = F.max_pool2d(x, 2, stride=2, padding=0)
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        #x = torch.nn.Dropout()(x)
        x = F.relu(self.fc2(x))
        #x = torch.nn.Dropout()(x)
        x = self.fc_out(x)
        x = F.softmax(x, dim=1)
        return x


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
                    xs.append(x)
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
    model = VGG16().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    model.train()

    xs, ts, paths = data_load('../Dataset/train/images/', hf=True, vf=True)

    # training
    mb = 16
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
        y = model(x)
        loss = loss_fn(torch.log(y), t)
        loss.backward()
        opt.step()
    
        pred = y.argmax(dim=1, keepdim=True)
        acc = pred.eq(t.view_as(pred)).sum().item() / mb
        
        print("iter >>", i+1, ',loss >>', loss.item(), ',accuracy >>', acc)

    torch.save(model.state_dict(), 'cnn.pt')

# test
def test():
    device = torch.device("cuda" if GPU else "cpu")
    model = VGG16().to(device)
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
