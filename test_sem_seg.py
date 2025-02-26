import os
import time
import string
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchinfo import summary
import open3d as o3d

# load all required modules for benchmarking: fully_supervised, semantic segmentation
from benchmark_semseg.utils.args import test_args
from benchmark_semseg.utils.calculate_mIoU import metric_evaluate
#from benchmark_semseg.utils.visual import visualize_results

def test_dataset(args):
  if args.dataset == 'touchstone3d':
    from touchstone3d_semseg.scripts.touchstone3d import Touchstone3DDataset
    testSet = Touchstone3DDataset(map_file='./touchstone3d_semseg/processed/meta/map.pkl', 
                                   npts=2048, mode='test', 
                                   data_path='./touchstone3d_semseg/processed/test/blocks', 
                                   pc_attribs='xyz', pc_augm=False, pc_augm_config=None)
   
    cls = testSet.CLASS_LABELS
   
  elif args.dataset == 's3dis':
    from s3dis_semseg.scripts.s3dis import S3DISDataset
    testSet = S3DISDataset(map_file='./s3dis_semseg/processed/meta/map.pkl', 
                                   npts=2048, mode='test', 
                                   data_path='./s3dis_semseg/processed/test/blocks', 
                                   pc_attribs='xyz', pc_augm=False, pc_augm_config=None)
    
  return testSet, cls 


def test(args, test_dataset, classes, logdir=None):
  class2label = {cls: i for i, cls in enumerate(classes)}
  seg_classes = class2label
  class2color = test_dataset.CLASS2COLOR
  seg_label_to_cat = {}
  for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

  testLoader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10,
                                                 pin_memory=True, drop_last=True)
  
  LOG.info(f'{len(test_dataset)} samples in test set.')
  classes_list = [string.capwords(i) for i in classes]
  LOG.info("Classes: {}".format(', '.join(map(str, classes_list))))

  # model
  if args.model == 'pointnet':
    from benchmark_semseg.model.pointnet_v1 import get_model
  elif args.model == 'pointnetplus':
    from benchmark_semseg.model.pointnetplus_v1 import get_model
  if args.model == 'pointnet' or args.model == 'pointnetplus':
    net = get_model(num_class=len(classes)).cuda()
    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['model_state_dict'])

    LOG.debug(summary(net, input_size=(args.batch_size, 3, args.npoint)))

    if args.visual:
        pred_dir = log_dir+'/pred'
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
    
  # evaluation
  with torch.no_grad():  
    pts_total = []
    pred_total = []
    gt_total = []
    
    net.eval()

    for i, (points, target, xyz_min, file_name) in tqdm(enumerate(testLoader), total=len(testLoader), smoothing=0.9):
      points, target = points.cuda(), target.long().cuda()
      points = points.transpose(2, 1)
      logits, trans_feat = net(points) # pred: [B, N, Cls]
      batch_size, num_points = logits.shape[0:2]
      logits = logits.contiguous().view(-1, len(classes)) # pred: [BxN, Cls]
      target = target.view(-1, 1)[:, 0] # target: [BxN]

      _, preds = torch.max(logits.detach(), dim=1, keepdim=False)

      pts_total.append(points.cpu().detach())
      pred_total.append(preds.cpu().detach())
      gt_total.append(target.detach())

      xyz_min = xyz_min.cpu().detach().numpy()
      if args.visual:
        points_vis = points.transpose(2,1).cpu().detach().numpy() 
        preds_vis = preds.view(batch_size, num_points).cpu().detach().numpy()
        for b in range(batch_size):
          pcd = o3d.geometry.PointCloud()
          points_vis[b] += xyz_min[b]
          pcd.points = o3d.utility.Vector3dVector(points_vis[b,:]*100)
          colors = np.zeros_like(points_vis[b,:])
          for n in range(num_points):
            colors[n,:] = np.asarray(class2color[preds_vis[b,n]])/255.0
          pcd.colors = o3d.utility.Vector3dVector(colors)
          o3d.io.write_point_cloud(pred_dir+'/'+file_name[b]+'.pcd', pcd)

    
    pts_total = torch.stack(pts_total, dim=0).view(-1, 3, args.npoint)
    pred_total = torch.stack(pred_total, dim=0).view(-1, args.npoint)
    gt_total = torch.stack(gt_total, dim=0).view(-1, args.npoint)
    # converting it to numpy for metric evaluation. 
    # GPU calculation is slower for some reason.
    pts_total = pts_total.numpy().astype(np.float32)
    pred_total = pred_total.numpy()
    gt_total = gt_total.cpu().numpy()
    LOG.debug(f'shape - total points in test set: {pts_total.shape}')
    LOG.debug(f'shape - total pred (labels) in test set: {pred_total.shape}')
    LOG.debug(f'shape - total GT (labels) in test set: {gt_total.shape}')

    acc, mIoU, iou_perclass, msiou, siou = metric_evaluate(pred_total, gt_total, len(classes))
    
    LOG.info(f'==========================[Test]==========================')
    LOG.info(f"msIoU: {msiou:.3f}")
    LOG.info(f"mIoU: {mIoU:.3f}")
    LOG.info(f"OA: {acc:.3f}")
    LOG.info(f"-----------------------iou per class-----------------------")  
    for i in range(len(iou_perclass)):
      LOG.info(f"{seg_label_to_cat[i]+''*(len(classes)+1 -len(classes[i]))}: {iou_perclass[i]:.3f}")


if __name__ == '__main__':
  import logging    # logging
  args = test_args()

  if args.log_dir == None:
    path = os.path.normpath(args.checkpoint)
    path = path.split(os.sep)
    log_dir = os.path.join(path[0], args.dataset, args.model)
  else:
    log_dir = os.path.join(args.log_dir, args.dataset, args.model)
  
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
      logging.FileHandler(log_dir+"/logs_test.log"),
      logging.StreamHandler()
    ])
  
  print(f'Saving logs to: {log_dir}')
  LOG = logging.getLogger(__name__)
  LOG.info(f'---------------------options---------------------')
  for k, v in sorted(opt.items()):
    LOG.info(f'{str(k)}:{str(v)}')
  LOG.info(f'-------------------------------------------------')
  testSet, classes = test_dataset(args)
  test(args, testSet, classes, logdir=log_dir)

  
  
    

       


  
