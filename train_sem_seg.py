import os
import time
import string
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# load all required modules for benchmarking: fully_supervised, semantic segmentation
from benchmark_semseg.utils.args import train_args
from benchmark_semseg.utils.inplace_relu import inplace_relu
from benchmark_semseg.utils.calculate_mIoU import metric_evaluate


def dataset(args):
  if args.dataset == 'touchstone3d': # dataset with sampling
    from touchstone3d_semseg.scripts.touchstone3d import Touchstone3DDataset
    pc_augm_config = {'scale': args.scale, 'rot': args.rot, 'mirror_prob': args.mirror_prob, 'jitter': args.jitter}
    trainSet = Touchstone3DDataset(map_file='./touchstone3d_semseg/processed/meta/map.pkl', 
                                   npts=2048, mode='train', 
                                   data_path='./touchstone3d_semseg/processed/train/blocks', 
                                   pc_attribs='xyz', pc_augm=True, pc_augm_config=pc_augm_config)
    valSet = Touchstone3DDataset(map_file='./touchstone3d_semseg/processed/meta/map.pkl', 
                                 npts=2048, mode='test', 
                                 data_path='./touchstone3d_semseg/processed/val/blocks', 
                                 pc_attribs='xyz')
    
    weights = [0.32905912, 0.17146748, 0.1963576,  0.1045766,  0.04412173, 0.04081506, 0.02961727, 
               0.00325412, 0.01689371, 0.00396157, 0.01667251, 0.00439292, 0.00840026, 0.00220573, 
               0.00348793, 0.00100488, 0.00216152, 0.02155003] # 18 classes
    weights = torch.FloatTensor(weights)
    cls = trainSet.CLASS_LABELS
   

  elif args.dataset == 's3dis':
    from s3dis_semseg.scripts.s3dis import S3DISDataset
    pc_augm_config = {'scale': args.scale, 'rot': args.rot, 'mirror_prob': args.mirror_prob, 'jitter': args.jitter}
    trainSet = S3DISDataset(map_file='./s3dis_semseg/processed/meta/map.pkl', 
                                   npts=2048, mode='train', 
                                   data_path='./s3dis_semseg/processed/train/blocks', 
                                   pc_attribs='xyz', pc_augm=True, pc_augm_config=pc_augm_config)
    valSet = S3DISDataset(map_file='./s3dis_semseg/processed/meta/map.pkl', 
                                 npts=2048, mode='test', 
                                 data_path='./s3dis_semseg/processed/val/blocks', 
                                 pc_attribs='xyz')
    
    weights = [0.21664207, 0.19724198, 0.25072926, 0.01911099, 0.01495163, 0.0147464, 
               0.06136107, 0.02295341, 0.04415964, 0.00410799, 0.03840661, 0.00965509, 0.10593385]  # 13 classes
    weights = torch.FloatTensor(weights)
    cls = trainSet.CLASS_LABELS
  
  return trainSet, valSet, cls, weights


def train(args, train_dataset, valid_dataset, classes, weights=None, logdir=None):
  WRITER = SummaryWriter(log_dir=log_dir)
  class2label = {cls: i for i, cls in enumerate(classes)}
  seg_classes = class2label
  seg_label_to_cat = {}
  for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

  trainLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10,
                                     pin_memory=True, drop_last=True,
                                     worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
  valLoader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10,
                                                 pin_memory=True, drop_last=True)
  
  LOG.info(f'{len(train_dataset)} samples in train set.')
  LOG.info(f'{len(valid_dataset)} samples in validation set.')
  classes_list = [string.capwords(i) for i in classes]
  LOG.info("Classes: {}".format(', '.join(map(str, classes_list))))
  
  # model
  if args.model == 'pointnet':
    from benchmark_semseg.model.pointnet_v1 import get_model, get_loss
  elif args.model == 'pointnetplus':
    from benchmark_semseg.model.pointnetplus_v1 import get_model, get_loss
  if args.model == 'pointnet' or args.model == 'pointnetplus':
    net = get_model(num_class=len(classes)).cuda()  
    criterion = get_loss().cuda()
    net.apply(inplace_relu)

    def weights_init(m):
      classname = m.__class__.__name__
      if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
      elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    
    net = net.apply(weights_init)

  if args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                 eps=1e-08, weight_decay=args.decay_rate)
  elif args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
  else:
    print('optimizer is not defined. Choices: [Adam, SGD]')

  def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
      m.momentum = momentum


  best_iou = 0.
  # training loop
  for epoch in range(0, args.epoch):
    if args.model == 'pointnet' or 'pointnetplus':
      lr = max(args.lr * (args.lr_decay ** (epoch // args.step_size)), args.lr_clip)
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr
      momentum = 0.1 * (0.5 ** (epoch // args.step_size))
      if momentum < 0.01: momentum = 0.01
      net = net.apply(lambda x: bn_momentum_adjust(x, momentum))

    running_loss = 0.   
    labelweights = np.zeros(len(classes))
    net.train() # model in training mode
    for batch_idx, (points, target) in tqdm(enumerate(trainLoader), total=len(trainLoader), smoothing=0.9):
      LOG.debug(f"Input tensor: {points.shape}")
      LOG.debug(f"Target tensor: {target.shape}")
      optimizer.zero_grad()
      points, target = points.float().cuda(), target.long().cuda()
      points = points.transpose(2, 1)
      logits, trans_feat = net(points)  # logits: [B, N, Cls]
      logits = logits.contiguous().view(-1, len(classes)) # pred: [BxN, Cls]
      target = target.view(-1, 1)[:, 0] # target: [BxN]
      loss = criterion(logits, target, trans_feat, weight=weights)
      loss.backward()
      optimizer.step()
      
      running_loss += loss.item() # accumulate loss for an epoch
      
      # calculate labelweights
      batch_label = target.cpu().data.numpy()
      tmp, _ = np.histogram(batch_label, range(len(classes) + 1))
      labelweights += tmp
    labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
    LOG.debug(f'class weights for train set: {labelweights}')

    WRITER.add_scalar('Train/loss', running_loss, epoch)
    LOG.info(f'====[Train] epoch: {epoch} - loss: {running_loss/args.batch_size:.3f}====')

    # evaluation
    with torch.no_grad():
      loss_val = 0.
      pred_total = []
      gt_total = []
      labelweights = np.zeros(len(classes))
      net.eval()
      for i, (points, target, xyz_min, file_name) in tqdm(enumerate(valLoader), total=len(valLoader), smoothing=0.9):
        points, target = points.float().cuda(), target.long().cuda()
        points = points.transpose(2, 1)
        logits, trans_feat = net(points) # pred: [B, N, Cls]
        logits = logits.contiguous().view(-1, len(classes)) # pred: [BxN, Cls]
        target = target.view(-1, 1)[:, 0] # target: [BxN]
        loss = criterion(logits, target, trans_feat, weight=weights)
        loss_val += loss

        _, preds = torch.max(logits.detach(), dim=1, keepdim=False)
        pred_total.append(preds.cpu().detach())
        gt_total.append(target.detach())

        # calculate labelweights
        batch_label = target.cpu().data.numpy()
        tmp, _ = np.histogram(batch_label, range(len(classes) + 1))
        labelweights += tmp
    labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))

    pred_total = torch.stack(pred_total, dim=0).view(-1, args.npoint)
    gt_total = torch.stack(gt_total, dim=0).view(-1, args.npoint)
    
    # converting it to numpy for metric evaluation. 
    # GPU calculation is slower for some reason.
    pred_total = pred_total.numpy()
    gt_total = gt_total.cpu().numpy()
  
    acc, mIoU, iou_perclass = metric_evaluate(pred_total, gt_total, len(classes))
    WRITER.add_scalar('Valid/loss', loss_val, epoch)
    WRITER.add_scalar('Valid/acc', acc, epoch)
    WRITER.add_scalar('Valid/mIoU', mIoU, epoch)  
    LOG.info(f'====[Valid] epoch: {epoch} - loss: {loss_val/args.batch_size:.3f} - mIoU: {mIoU:.3f} - OA: {acc:.3f}====')

    LOG.info(f"---------------------iou per class---------------------")  
    for i in range(len(iou_perclass)):
      LOG.info(f"{seg_label_to_cat[i]+''*(len(classes)+1 -len(classes[i]))}: {iou_perclass[i]:.3f} - weights: {labelweights[i]:.3f}")

    # save model
    if mIoU >= best_iou:
      best_iou = mIoU
      LOG.info(f'Saving net... (in epoch, {epoch})')
      save_model = os.path.join(log_dir, 'model.pth')
      state = {
          'epoch': epoch,
          'class_avg_iou': mIoU,
          'model_state_dict': net.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
      }
      torch.save(state, save_model)

    
if __name__ == '__main__':
  import logging    # logging
  args = train_args()
  log_dir = os.path.join(args.log_dir)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  opt = vars(args)
  # logging config
  if args.debug:
    LEVEL = logging.DEBUG
  else: 
    LEVEL = logging.INFO
  logging.basicConfig(format='%(message)s', level=LEVEL,
    handlers=[
      logging.FileHandler(log_dir+"/logs.log"),
      logging.StreamHandler()
    ])
  LOG = logging.getLogger(__name__)
  LOG.info(f'---------------------options---------------------')
  for k, v in sorted(opt.items()):
    LOG.info(f'{str(k)}:{str(v)}')
  LOG.info(f'-------------------------------------------------')
  trainSet, valSet, classes, cls_W = dataset(args)
  cls_W = cls_W.cuda()
  if args.weighted_loss:
    train(args, trainSet, valSet, classes, cls_W, logdir=log_dir)
  else:
    train(args, trainSet, valSet, classes, None, logdir=log_dir)

