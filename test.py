import os
import argparse
import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable

from utils import make_dataset, edge_compute

parser = argparse.ArgumentParser()
parser.add_argument('--network', default='GCANet')
parser.add_argument('--task', default='dehaze', help='dehaze | derain')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--indir', default='examples/')
parser.add_argument('--outdir', default='output')
opt = parser.parse_args()
assert opt.task in ['dehaze', 'derain']
## forget to regress the residue for deraining by mistake,
## which should be able to produce better results
opt.only_residual = opt.task == 'dehaze'  
opt.model = 'models/wacv_gcanet_%s.pth' % opt.task 
opt.use_cuda = opt.gpu_id >= 0
if not os.path.exists(opt.outdir):
    os.makedirs(opt.outdir)
test_img_paths = make_dataset(opt.indir)

if opt.network == 'GCANet':
    from GCANet import GCANet
    net = GCANet(in_c=4, out_c=3, only_residual=opt.only_residual)
else:
    print('network structure %s not supported' % opt.network)
    raise ValueError

if opt.use_cuda:
    torch.cuda.set_device(opt.gpu_id)
    net.cuda()
else:
    net.float()

net.load_state_dict(torch.load(opt.model, map_location='cpu'))
net.eval()

for img_path in test_img_paths:
    img = Image.open(img_path).convert('RGB')
    im_w, im_h = img.size
    if im_w % 4 != 0 or im_h % 4 != 0:
        img = img.resize((int(im_w // 4 * 4), int(im_h // 4 * 4))) 
    img = np.array(img).astype('float')
    img_data = torch.from_numpy(img.transpose((2, 0, 1))).float()
    edge_data = edge_compute(img_data)
    in_data = torch.cat((img_data, edge_data), dim=0).unsqueeze(0) - 128 
    in_data = in_data.cuda() if opt.use_cuda else in_data.float()
    with torch.no_grad():
        pred = net(Variable(in_data))
    if opt.only_residual:
        out_img_data = (pred.data[0].cpu().float() + img_data).round().clamp(0, 255)
    else:
        out_img_data = pred.data[0].cpu().float().round().clamp(0, 255)
    out_img = Image.fromarray(out_img_data.numpy().astype(np.uint8).transpose(1, 2, 0))
    out_img.save(os.path.join(opt.outdir, os.path.splitext(os.path.basename(img_path))[0] + '_%s.png' % opt.task))





