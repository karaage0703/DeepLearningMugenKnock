import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
from glob import glob

num_classes = 2
img_height, img_width = 64, 64
GPU = False
torch.manual_seed(0)

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


# get train data
def data_load():
    xs = np.ndarray((0, img_height, img_width, 3))
    ts = np.ndarray((0))
    
    for dir_path in glob('../../Dataset/train/images/*'):
        for path in glob(dir_path + '/*'):
            x = cv2.imread(path)
            x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
            x /= 255.
            xs = np.r_[xs, x[None, ...]]

            t = np.zeros((1))
            if 'akahara' in path:
                t = np.array((0))
            elif 'madara' in path:
                t = np.array((1))
            ts = np.r_[ts, t]

    xs = xs.transpose(0,3,1,2)

    return xs, ts


# train
def train():
    # GPU
    device = torch.device("cuda" if GPU else "cpu")

    # model
    model = Mynet().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model.train()

    xs, ts = data_load()

    # training
    mb = 8
    mbi = 0
    train_ind = np.arange(len(xs))
    np.random.seed(0)
    np.random.shuffle(train_ind)
    
    for i in range(100):
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
        y = F.log_softmax(y, dim=1)
        loss = torch.nn.CrossEntropyLoss()(y, t)
        loss.backward()
        opt.step()
    
        pred = y.argmax(dim=1, keepdim=True)
        acc = pred.eq(t.view_as(pred)).sum().item() / mb
        
        print("iter >>", i+1, ',loss >>', loss.item(), ',accuracy >>', acc)

    torch.save(model.state_dict(), 'cnn.pt')

# test
def test():
    device = torch.device("cuda" if GPU else "cpu")
    model = Mynet().to(device)
    model.eval()
    model.load_state_dict(torch.load('cnn.pt'))

    for dir_path in glob('../../Dataset/test/images/*'):
        for path in glob(dir_path + '/*'):
            x = cv2.imread(path)
            x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
            x /= 255.
            x = x.transpose(2, 0, 1)
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
