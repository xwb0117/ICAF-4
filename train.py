import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
import icaf_util
import h5py
import torch.multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import utils
from data_utils.Art_DataLoader import ArtImageDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def get_Anchor_Candidates():
    quat1 = utils.get_anchor([0,1,0],[0,0,1])
    quat2 = utils.get_anchor([1,0,0],[0,0,1])
    quat3 = utils.get_anchor([1,0,0],[0,1,0])
    quats = np.concatenate([quat1, quat2, quat3], axis=0)
    anchor_quats = np.array(quats)
    anchor_quats = torch.tensor(anchor_quats, dtype=torch.float32)
    anchor_quats = anchor_quats.float().cuda()
    return anchor_quats

def calculate_angle(q1, q2):
    q1_expanded = q1.expand_as(q2)
    dot_product = torch.sum(q1_expanded * q2, dim=1)
    angle = torch.acos(dot_product.clamp(-1.0, 1.0))
    return angle

def find_closest_anchor(input_quaternions, anchor_quaternions):
    n = input_quaternions.size(0)
    m = input_quaternions.size(1)
    segmentation_tensor = torch.zeros((n,m), dtype=torch.int64)
    for i in range(n):
        for j in range(m):
            angles = torch.norm((input_quaternions[i][j] - anchor_quaternions[i][j]), dim=-1)
            min_index = torch.argmin(angles)
            segmentation_tensor[i][j] = min_index

    return segmentation_tensor

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_part_seg', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=251, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--category', type=str, default='laptop', help='category of articulation')
    parser.add_argument('--num_part', type=int, default=2, help='number of articulation part')
    parser.add_argument('--L', type=int, default=108, help='length of anchor candidates')
    parser.add_argument('--save_results', type=bool, default=False, help='save results')

    return parser.parse_args()

def main(args):
    mp.set_start_method('spawn')
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('Art_ICAF-4_log')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    result_dir = exp_dir.joinpath('results/')
    checkpoints_dir.mkdir(exist_ok=True)
    result_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = '/mnt/7797b2ec-a944-4795-abb2-f472a7fc833e/Data/ArtImage'

    TRAIN_DATASET = ArtImageDataset(root=root, npoints=args.npoint, split='train', normal_channel=args.normal, cat=args.category)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=20, drop_last=True)
    TEST_DATASET = ArtImageDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal, cat=args.category)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=20)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    num_part = args.num_part

    anchor_candidates = get_Anchor_Candidates()

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    model = MODEL.get_model(num_part, args.L).cuda()
    model.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        model = model.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_result = np.inf
    best_model = None

    for epoch in range(start_epoch, args.epoch):
        AvgRecorder = MODEL.AvgRecorder()
        epoch_loss = {
            'total_loss': AvgRecorder
        }

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        model = model.apply(lambda x: bn_momentum_adjust(x, momentum))
        model = model.train()

        '''learning one epoch'''
        log_string(f'>>>>>>>>>>>>>>>> Start Train >>>>>>>>>>>>>>>>')
        for i, (points, gt_dict, id) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            device = torch.device("cuda:0")
            points = points.to(device)

            quats_per_point = gt_dict['quats_per_point']
            expanded_quats_per_point = (quats_per_point.unsqueeze(2).expand(quats_per_point.shape[0], quats_per_point.shape[1], 1, 4)).cuda()  # nx1x4
            expanded_anchor_candidates = anchor_candidates.unsqueeze(0).unsqueeze(0).repeat(quats_per_point.shape[0], quats_per_point.shape[1], 1, 1)
            D_quats = expanded_quats_per_point - expanded_anchor_candidates  # nxLx4
            anchor_seg_per_point = find_closest_anchor(expanded_quats_per_point, expanded_anchor_candidates).cuda()  # nxLx1

            gt = {}
            gt['D_quats_per_point'] = D_quats
            gt['anchor_candidates_per_point'] = expanded_anchor_candidates
            gt['anchor_seg_per_point'] = anchor_seg_per_point
            for k, v in gt_dict.items():
                gt[k] = v.to(device)

            pred = model(points)

            loss_dict = model.losses(pred, gt, args.L)

            loss = torch.tensor(0.0, device=device)

            loss_weight = model.loss_weight()
            # use different loss weight to calculate the final loss
            for k, v in loss_dict.items():
                if k not in loss_weight:
                    raise ValueError(f"No loss weight for {k}")
                loss += loss_weight[k] * v

            # Used to calculate the avg loss
            for k, v in loss_dict.items():
                if k not in epoch_loss.keys():
                    epoch_loss[k] = MODEL.AvgRecorder()
                epoch_loss[k].update(v)
            epoch_loss['total_loss'].update(loss)

            loss.backward()
            optimizer.step()

        loss_log = ''
        for k, v in epoch_loss.items():
            loss_log += '{}: {:.5f}  '.format(k, v.avg)

        log_string('Epoch: {}/{} Loss: {}'.format(epoch+1, args.epoch, loss_log))


        '''val'''
        log_string(f'>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
        with torch.no_grad():
            model = model.eval()
            val_error = {
                'total_loss': AvgRecorder
            }

            if args.save_results:
                inference_path = str(result_dir) + '/results.h5'
                test_result = h5py.File(inference_path, "w")

            for i, (points, gt_dict, id) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                device = torch.device("cuda:0")
                points = points.to(device)

                quats_per_point = gt_dict['quats_per_point']
                expanded_quats_per_point = (quats_per_point.unsqueeze(2).expand(quats_per_point.shape[0], quats_per_point.shape[1], 1,4)).cuda()  # nx1x4
                expanded_anchor_candidates = anchor_candidates.unsqueeze(0).unsqueeze(0).repeat(quats_per_point.shape[0], quats_per_point.shape[1], 1, 1)
                D_quats = expanded_quats_per_point - expanded_anchor_candidates  # nxLx4
                anchor_seg_per_point = find_closest_anchor(expanded_quats_per_point,expanded_anchor_candidates).cuda()  # nxLx1

                gt = {}
                gt['D_quats_per_point'] = D_quats
                gt['anchor_candidates_per_point'] = expanded_anchor_candidates
                gt['anchor_seg_per_point'] = anchor_seg_per_point
                for k, v in gt_dict.items():
                    gt[k] = v.to(device)

                pred = model(points)
                if args.save_results:
                    icaf_util.save_grasp_results(test_result, pred, points, gt, id)

                loss_dict = model.losses(pred, gt, args.L)

                loss = torch.tensor(0.0, device=device)

                loss_weight = model.loss_weight()
                # use different loss weight to calculate the final loss
                for k, v in loss_dict.items():
                    if k not in loss_weight:
                        raise ValueError(f"No loss weight for {k}")
                    loss += loss_weight[k] * v

                # Used to calculate the avg loss
                for k, v in loss_dict.items():
                    if k not in val_error.keys():
                        val_error[k] = MODEL.AvgRecorder()
                    val_error[k].update(v)
                val_error['total_loss'].update(loss)

        loss_log = ''
        for k, v in val_error.items():
            loss_log += '{}: {:.5f}  '.format(k, v.avg)

        log_string('Eval Epoch: {}/{} Loss: {}'.format(epoch+1, args.epoch, loss_log))

        if args.save_results:
            test_result.close()

        #save model
        if best_model is None or val_error["total_loss"].avg < best_result:
            logger.info('Save model...')
            logger.info('best_result: {}'.format(best_result))
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            best_result = val_error["total_loss"].avg
            torch.save(state, savepath)
            log_string('Saving model....')


if __name__ == '__main__':
    args = parse_args()
    main(args)