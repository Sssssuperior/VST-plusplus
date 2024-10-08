import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from dataset import get_loader
import transforms as trans
from torchvision import transforms
import time
from Models.ImageDepthNet import ImageDepthNet
from torch.utils import data
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import numpy as np
import os


def save_loss(save_dir, dataset, time):
    fh = open(save_dir, 'a') 
    fh.write('dataset' + dataset + 'time' + str(time) + '\n')
    fh.write('\n')
    fh.close()
    

def test_net(args):

    cudnn.benchmark = True

    net = ImageDepthNet(args)
    net.cuda()
    net.eval()

    # load model (multi-gpu)
    model_path = args.save_model_dir + 'RGB_VST.pth'
    state_dict = torch.load(model_path)
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    net.load_state_dict(new_state_dict)
    print('Model loaded from {}'.format(model_path))

    # load model
    # net.load_state_dict(torch.load(model_path))
    # model_dict = net.state_dict()
    # print('Model loaded from {}'.format(model_path))

    test_paths = args.test_paths.split('+')
    for test_dir_img in test_paths:

        test_dataset = get_loader(test_dir_img, args.data_root, args.img_size, mode='test')

        test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1)
        print('''
                   Starting testing:
                       dataset: {}
                       Testing size: {}
                   '''.format(test_dir_img.split('/')[0], len(test_loader.dataset)))

        time_list = []
        for i, data_batch in enumerate(test_loader):
            images, image_w, image_h, image_path = data_batch
            images = Variable(images.cuda())
            #flops = FlopCountAnalysis(net, (images))
            #m = flops.total()
            #print(parameter_count_table(net))
            starts = time.time()
            outputs_saliency, outputs_contour, outputs_saliency_s, outputs_contour_s = net(images)
            ends = time.time()
            time_use = ends - starts
            time_list.append(time_use)

            mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency
            mask_1_16_s, mask_1_8_s, mask_1_4_s, mask_1_1_s = outputs_saliency_s

            image_w, image_h = int(image_w[0]), int(image_h[0])

            output_s = F.sigmoid(mask_1_1)
            output_s_s = F.sigmoid(mask_1_1_s)

            output_s = output_s.data.cpu().squeeze(0)
            output_s_s = output_s_s.data.cpu().squeeze(0)

            transform = trans.Compose([
                transforms.ToPILImage(),
                trans.Scale((image_w, image_h))
            ])
            output_s = transform(output_s)
            output_s_s = transform(output_s_s)

            dataset = test_dir_img.split('/')[0]
            filename = image_path[0].split('/')[-1].split('.')[0]

            # save saliency maps
            save_test_path1 = args.save_test_path_root + dataset +  '/RGB_VST_fenge/'
            save_test_path2 = args.save_test_path_root + dataset +  '/RGB_VST_yuanshi/'
            if not os.path.exists(save_test_path1):
                os.makedirs(save_test_path1)
            if not os.path.exists(save_test_path2):
                os.makedirs(save_test_path2)
            output_s.save(os.path.join(save_test_path1, filename + '.png'))
            output_s_s.save(os.path.join(save_test_path2, filename + '.png'))

        print('dataset:{}, cost:{}'.format(test_dir_img.split('/')[0], np.mean(time_list) * 1000))
        save_timedir = './time.txt'
        save_loss(save_timedir, test_dir_img.split('/')[0], np.mean(time_list) * 1000)




