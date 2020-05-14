import torch
import torch.nn.functional as F
import argparse
import cv2
import numpy as np
from glob import glob
import copy
from tqdm import tqdm


# get train data
def data_load(cfg, path, hf=False, vf=False, rot=False):
    if (rot == 0) and (rot != False):
        raise Exception('invalid rot >> ', rot, 'should be [1, 359] or False')

    paths = []
    ts = []
    
    data_num = 0
    for dir_path in glob(path + '/*'):
        data_num += len(glob(dir_path + "/*"))
            
    print('Dataset >>', path)
    print(' - Found data num >>', data_num)
    print(' - Horizontal >>', hf)
    print(' - Vertical >>', vf)
    print(' - Rotation >>', rot)
    pbar = tqdm(total = data_num)
    
    for dir_path in glob(path + '/*'):
        for path in glob(dir_path + '/*'):
            for i, cls in enumerate(cfg.CLASS_LABEL):
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
    
    print('all data num >>', len(paths))
    print('dataset was completely loaded')
    print('--')

    return np.array(paths), np.array(ts)


def get_image(cfg, infos):
    xs = []
    
    for info in infos:
        path = info['path']
        hf = info['hf']
        vf = info['vf']
        rot = info['rot']
        x = cv2.imread(path)

        # resize
        x = cv2.resize(x, (cfg.INPUT_WIDTH, cfg.INPUT_HEIGHT)).astype(np.float32)
        
        # channel BGR -> Gray
        if cfg.INPUT_CHANNEL == 1:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = np.expand_dims(x, axis=-1)

        # channel BGR -> RGB
        if cfg.INPUT_CHANNEL == 3:
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
        x = _x[tx:tx+_w, ty:ty+_h]

        xs.append(x)
                
    xs = np.array(xs, dtype=np.float32)
    xs = np.transpose(xs, (0,3,1,2))
    
    return xs



# train
def train(cfg, _model):
    print('-' * 20)
    print('train function')
    print('-' * 20)
    # model
    model = _model.to(cfg.DEVICE)
    opt = cfg.OPTIMIZER(model.parameters(), lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM)
    model.train()

    paths, ts = data_load(cfg, cfg.TRAIN.DATA_PATH, hf=cfg.TRAIN.DATA_HORIZONTAL_FLIP, vf=cfg.TRAIN.DATA_VERTICAL_FLIP, rot=cfg.TRAIN.DATA_ROTATION)

    # training
    mbi = 0
    data_N = len(paths)
    train_ind = np.arange(data_N)
    np.random.seed(0)
    np.random.shuffle(train_ind)

    loss_fn = cfg.LOSS_FUNCTION

    print('training start')
    progres_bar = ''
    
    for i in range(cfg.ITERATION):
        if mbi + cfg.MINIBATCH > data_N:
            mb_ind = copy.copy(train_ind)[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[ : (cfg.MINIBATCH - (data_N - mbi))]))
        else:
            mb_ind = train_ind[mbi : mbi + cfg.MINIBATCH]
            mbi += cfg.MINIBATCH

        x = torch.tensor(get_image(cfg, paths[mb_ind]), dtype=torch.float).to(cfg.DEVICE)
        t = torch.tensor(ts[mb_ind], dtype=torch.long).to(cfg.DEVICE)

        opt.zero_grad()
        y = model(x)
        #y = F.log_softmax(y, dim=1)
        loss = loss_fn(torch.log(y), t)
        
        loss.backward()
        opt.step()
    
        pred = y.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(t.view_as(pred)).sum().item() / cfg.MINIBATCH

        progres_bar += '|'
        print('\r' + progres_bar, end='')

        if (i + 1) % 10 == 0:
            progres_bar += str(i + 1)
            print('\r' + progres_bar, end='')

        # display training state
        if (i + 1) % cfg.TRAIN.DISPAY_ITERATION_INTERVAL == 0:
            print('\r' + ' ' * len(progres_bar), end='')
            print('\rIter : {} , Loss : {:.4f} , Accuracy : {:.4f}'.format(i+1, loss.item(), accuracy))
            progres_bar = ''

        # save parameters
        if (cfg.MODEL_SAVE_INTERVAL != False) and ((i + 1) % cfg.MODEL_SAVE_INTERVAL == 0):
            save_path = cfg.MODEL_SAVE_PATH.format('iter{}'.format(i + 1))
            torch.save(model.state_dict(), save_path)
            print('model was saved to >>', save_path)

    save_path = cfg.MODEL_SAVE_PATH.format('final')
    torch.save(model.state_dict(), save_path)
    print('model was saved to >>', save_path)

# test
def test(cfg, _model):
    print('-' * 20)
    print('test function')
    print('-' * 20)
    model = _model.to(cfg.DEVICE)
    model.load_state_dict(torch.load(cfg.TEST.MODEL_PATH, map_location=torch.device(cfg.DEVICE)))
    model.eval()

    print('model loaded >>', cfg.TEST.MODEL_PATH)

    paths, ts = data_load(cfg, cfg.TEST.DATA_PATH, hf=False, vf=False, rot=False)

    Test_Num = len(paths)

    print('test start')

    with torch.no_grad():
        for i in range(0, Test_Num, cfg.TEST.MINIBATCH):
            test_inds = np.arange(i, min(i + cfg.TEST.MINIBATCH, Test_Num))
            path = paths[test_inds]
            x = get_image(cfg, path)
            t = ts[test_inds]
            
            x = torch.tensor(x, dtype=torch.float).to(cfg.DEVICE)
            
            preds = model(x)

            for j in range(len(preds)):
                pred = preds.detach().cpu().numpy()[j]
                print('Data : {}, probabilities >> {}'.format(path[j], pred))
    

def arg_parse():
    parser = argparse.ArgumentParser(description='CNN implemented with Keras')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    args = parser.parse_args()
    return args


def main(cfg, model):
    args = arg_parse()

    if args.train:
        train(cfg, model)
    if args.test:
        test(cfg, model)

    if not (args.train or args.test):
        print("please select train or test flag")
        print("train: python main.py --train")
        print("test:  python main.py --test")
        print("both:  python main.py --train --test")
