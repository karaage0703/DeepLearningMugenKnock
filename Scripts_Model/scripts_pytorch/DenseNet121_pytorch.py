import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
from glob import glob
import copy

# class config
class_label = ['akahara', 'madara']
class_N = len(class_label)

# config
img_height, img_width = 96, 96
channel = 3

# GPU
GPU = False
device = torch.device("cuda" if GPU and torch.cuda.is_available() else "cpu")

# other
model_path = 'DenseNet121.pt'

# random seed
torch.manual_seed(0)
            
class Block(torch.nn.Module):
    def __init__(self, first_dim, k=32, L=6):
        self.L = L
        
        super(Block, self).__init__()

        self.blocks = torch.nn.ModuleList()
        
        self.blocks.append(torch.nn.Sequential(
                torch.nn.BatchNorm2d(first_dim),
                torch.nn.ReLU(),
                torch.nn.Conv2d(first_dim, k, kernel_size=1, padding=0, stride=1),
                torch.nn.BatchNorm2d(k),
                torch.nn.ReLU(),
                torch.nn.Conv2d(k, k, kernel_size=3, padding=1, stride=1),
            ))
        
        for i in range(1, L):
            self.blocks.append(torch.nn.Sequential(
                torch.nn.BatchNorm2d(k * i + first_dim),
                torch.nn.ReLU(),
                torch.nn.Conv2d(k * i + first_dim, k, kernel_size=1, padding=0, stride=1),
                torch.nn.BatchNorm2d(k),
                torch.nn.ReLU(),
                torch.nn.Conv2d(k, k, kernel_size=3, padding=1, stride=1),
            ))
        
        
    def forward(self, x):
        xs = [None for _ in range(self.L + 1)]
        xs[0] = x
        xs[1] = self.blocks[0](x)
        
        for i in range(1, self.L):
            x_in = xs[i]
            for j in range(i):
                x_in = torch.cat([x_in, xs[j]], dim=1)
            x = self.blocks[i](x_in)
            xs[i + 1] = x
                
        x = xs[0]
        for i in range(1, (self.L + 1)):
            x = torch.cat([x, xs[i]], dim=1)

        return x

        

class DenseNet121(torch.nn.Module):
    def __init__(self):
        super(DenseNet121, self).__init__()

        k = 32
        theta = 0.5
        self.bn1 = torch.nn.BatchNorm2d(channel)
        self.conv1 = torch.nn.Conv2d(channel, k * 2, kernel_size=7, padding=3, stride=2)
        
        # Dense block1
        block1_L = 6
        block1_dim = int(k * block1_L * theta)
        
        self.block1 = Block(first_dim = k * 2, L = block1_L)
        
        # Transition layer1
        self.transition1 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(k * block1_L + k * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(k * block1_L + k * 2, block1_dim, kernel_size=1, padding=0, stride=1),
            torch.nn.AvgPool2d(2, stride=2, padding=0)
        )
    
        # Dense block2
        block2_L = 12
        block2_dim = int(k * block2_L * theta)
        
        self.block2 = Block(first_dim = block1_dim, L = block2_L)

        # Transition layer2        
        self.transition2 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(k * block2_L + block1_dim),
            torch.nn.ReLU(),
            torch.nn.Conv2d(k * block2_L + block1_dim, block2_dim, kernel_size=1, padding=0, stride=1),
            torch.nn.AvgPool2d(2, stride=2, padding=0)
        )
        
        # Dense block3
        block3_L = 24
        block3_dim = int(k * block3_L * theta)
        
        self.block3 = Block(first_dim = block2_dim, L = block3_L)
        
        # Transition layer3
        self.transition3 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(k * block3_L + block2_dim),
            torch.nn.ReLU(),
            torch.nn.Conv2d(k * block3_L + block2_dim, block3_dim, kernel_size=1, padding=0, stride=1),
            torch.nn.AvgPool2d(2, stride=2, padding=0)
        )
        
        # Dense block4
        block4_L = 16
        self.block4 = Block(first_dim = block3_dim, L = block4_L)
        
        self.linear = torch.nn.Linear(k * block4_L + block3_dim, num_classes)
        
        
    def forward(self, x):
        # Entry flow
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv1(x)
        
        x = F.max_pool2d(x, 3, padding=1, stride=2)
        
        x = self.block1(x)
        
        x = self.transition1(x)
        
        x = self.block2(x)
        
        x = self.transition2(x)
        
        x = self.block3(x)
        
        x = self.transition3(x)
        
        x = self.block4(x)

        x = F.avg_pool2d(x, [img_height // 32, img_width // 32], padding=0, stride=1)
        x = x.view(list(x.size())[0], -1)
        x = self.linear(x)
        x = F.softmax(x, dim=1)
        
        return x

# get train data
def data_load(path, hf=False, vf=False, rot=False):
    if rot == 0:
        raise Exception('invalid rot >> ', rot, 'should be [1, 359] or False')

    paths = []
    ts = []
    
    data_num = 0
    for dir_path in glob(path + '/*'):
        data_num += len(glob(dir_path + "/*"))
            
    pbar = tqdm(total = data_num)
    
    for dir_path in glob(path + '/*'):
        for path in glob(dir_path + '/*'):
            for i, cls in enumerate(class_label):
                if cls in path:
                    t = i

            paths.append({'path': path, 'hf': False, 'vf': False, 'rot': 0})
            ts.append(t)

            # horizontal flip
            if hf:
                paths.append({'path': path, 'hf': True, 'vf': False, 'rot': 0})
                ts.append(t)
            # vertical flip
            if vf:
                paths.append({'path': path, 'hf': False, 'vf': True, 'rot': 0})
                ts.append(t)
            # horizontal and vertical flip
            if hf and vf:
                paths.append({'path': path, 'hf': True, 'vf': True, 'rot': 0})
                ts.append(t)
            # rotation
            if rot is not False:
                angle = rot
                while angle < 360:
                    paths.append({'path': path, 'hf': False, 'vf': False, 'rot': rot})
                    angle += rot
                    ts.append(t)
                
            pbar.update(1)
                    
    pbar.close()
    
    return np.array(paths), np.array(ts)

def get_image(infos):
    xs = []
    
    for info in infos:
        path = info['path']
        hf = info['hf']
        vf = info['vf']
        rot = info['rot']
        x = cv2.imread(path)

        # resize
        x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
        
        # channel BGR -> Gray
        if channel == 1:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = np.expand_dims(x, axis=-1)

        # channel BGR -> RGB
        if channel == 3:
            x = x[..., ::-1]

        # normalization [0, 255] -> [-1, 1]
        x = x / 127.5 - 1

        # horizontal flip
        if hf:
            x = x[:, ::-1]

        # vertical flip
        if vf:
            x = x[::-1]

        # rotation
        scale = 1
        _h, _w, _c = x.shape
        max_side = max(_h, _w)
        tmp = np.zeros((max_side, max_side, _c))
        tx = int((max_side - _w) / 2)
        ty = int((max_side - _h) / 2)
        tmp[ty: ty+_h, tx: tx+_w] = x.copy()
        M = cv2.getRotationMatrix2D((max_side / 2, max_side / 2), rot, scale)
        _x = cv2.warpAffine(tmp, M, (max_side, max_side))
        _x = _x[tx:tx+_w, ty:ty+_h]

        xs.append(x)
                
    xs = np.array(xs, dtype=np.float32)
    xs = np.transpose(xs, (0,3,1,2))
    
    return xs


# train
def train():
    # model
    model = DensetNet121().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()

    paths, ts = data_load('../Dataset/train/images/', hf=True, vf=True, rot=1)

    # training
    mb = 32
    mbi = 0
    data_N = len(paths)
    train_ind = np.arange(data_N)
    np.random.seed(0)
    np.random.shuffle(train_ind)

    loss_fn = torch.nn.NLLLoss()
    
    for i in range(500):
        if mbi + mb > data_N:
            mb_ind = copy.copy(train_ind)[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb - (data_N - mbi))]))
        else:
            mb_ind = train_ind[mbi : mbi + mb]
            mbi += mb

        x = torch.tensor(get_image(paths[mb_ind]), dtype=torch.float).to(device)
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

    torch.save(model.state_dict(), model_path)

# test
def test():
    model = DensetNet121().to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    paths, ts = data_load('../Dataset/test/images/', hf=False, vf=False, rot=False)

    with torch.no_grad():
        for i in range(len(paths)):
            path = paths[i]
            x = get_image(path)
            t = ts[i]
            
            x = np.expand_dims(x, axis=0)
            x = torch.tensor(x, dtype=torch.float).to(device)
            
            pred = model(x)
            pred = F.softmax(pred, dim=1).detach().cpu().numpy()[0]
        
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
