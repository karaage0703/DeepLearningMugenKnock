from glob import glob
from tqdm import tqdm
import numpy as np
import cv2

# get train data
def data_load(cfg):
    path = cfg.TRAIN.DATA_PATH
    hf = cfg.TRAIN.DATA_HORIZONTAL_FLIP
    vf = cfg.TRAIN.DATA_VERTICAL_FLIP
    rot = cfg.TRAIN.DATA_ROTATION

    if (rot == 0) and (rot != False):
        raise Exception('invalid rot >> ', rot, 'should be [1, 359] or False')

    paths = []
    paths_gt = []
    
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
            
            gt_path = path.replace("images", "seg_images").replace(".jpg", ".png")
            paths_gt.append({'path': gt_path, 'hf': False, 'vf': False, 'rot': 0})

            # horizontal flip
            if hf:
                paths.append({'path': path, 'hf': True, 'vf': False, 'rot': 0})
                paths_gt.append({'path': gt_path, 'hf': True, 'vf': False, 'rot': 0})
            # vertical flip
            if vf:
                paths.append({'path': path, 'hf': False, 'vf': True, 'rot': 0})
                paths_gt.append({'path': gt_path, 'hf': False, 'vf': True, 'rot': 0})
            # horizontal and vertical flip
            if hf and vf:
                paths.append({'path': path, 'hf': True, 'vf': True, 'rot': 0})
                paths_gt.append({'path': gt_path, 'hf': True, 'vf': True, 'rot': 0})
            # rotation
            if rot is not False:
                angle = rot
                while angle < 360:
                    paths.append({'path': path, 'hf': False, 'vf': False, 'rot': rot})
                    paths_gt.append({'path': gt_path, 'hf': False, 'vf': False, 'rot': rot})
                    angle += rot
                
            pbar.update(1)
                    
    pbar.close()

    print('all data num >>', len(paths))
    print('dataset was completely loaded')
    print('--')

    
    return {'paths' : np.array(paths), 'paths_gt' : np.array(paths_gt)}


def get_image(infos, cfg, mode):
    xs = []
    
    for info in infos:
        path = info['path']
        hf = info['hf']
        vf = info['vf']
        rot = info['rot']
        x = cv2.imread(path)

        # resize
        x = cv2.resize(x, (cfg.OUTPUT_WIDTH, cfg.OUTPUT_HEIGHT)).astype(np.float32)
        
        # channel BGR -> Gray
        if mode == 'GRAY':
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = np.expand_dims(x, axis=-1)
        elif mode == 'EDGE':
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = cv2.Canny(x, 100, 150)
            x = np.expand_dims(x, axis=-1)

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

        if mode == 'CLASS_LABEL':
            _x = x
            x = np.zeros((cfg.OUTPUT_HEIGHT, cfg.OUTPUT_WIDTH, cfg.CLASS_NUM), dtype=np.int)

            for i, (_, vs) in enumerate(cfg.CLASS_LABEL.items()):
                ind = (_x[..., 0] == vs[0]) * (_x[..., 1] == vs[1]) * (_x[..., 2] == vs[2])
                x[..., i][ind] = 1

        else:
            # normalization [0, 255] -> [-1, 1]
            x = x / 127.5 - 1

            # channel BGR -> RGB
            if mode in ['RGB']:
                x = x[..., ::-1]

        xs.append(x)
                
    xs = np.array(xs, dtype=np.float32)

    return xs