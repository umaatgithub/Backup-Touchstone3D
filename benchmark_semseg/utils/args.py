import configargparse

def train_args():
  parser = configargparse.ArgParser(default_config_files=['pointnetplus.cfg','pointnet.cfg', 'dgcnn.cfg'])
  parser.add('-c', '--cfg', required=True, is_config_file=True, help='config file path')
  parser.add('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
  parser.add('--dataset', type=str, default='touchstone3d', help='dataset name [touchstone, s3dis]')
  parser.add('--model', type=str, default='pointnet', help='model name [pointnet, pointnetplus, dgcnn, kpconv]')
  parser.add('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
  parser.add('--epoch', default=10000, type=int, help='Epoch to run [default: 32]')
  parser.add('--lr', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
  parser.add('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
  parser.add('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
  parser.add('--npoint', type=int, default=2048, help='Point Number [default: 2048]')
  parser.add('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
  parser.add('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
  parser.add('--lr_clip', type=float, default=1e-5, help='clipping for learning rate [default: 1e-5s]')
  parser.add('--weighted_loss', type=bool, default=False, help='Weighted loss with class weights [default: True]')
  # augmentation
  parser.add('--scale', type=int, default=0, help='scale: [0, 1]')
  parser.add('--rot', type=int, default=1, help='rot: [0, 1]')
  parser.add('--mirror_prob', type=int, default=0, help='mirror_prob: [0, 1]')
  parser.add('--jitter', type=int, default=1, help='jitter: [0, 1]')
  # debug
  parser.add('--debug', type=bool, default=False, help='allows to use log.debug')
  # log directory to save model and log
  parser.add('--log_dir', type=str, default=None, help='Log path [default: None]')

  return parser.parse_args()



def test_args():
  parser = configargparse.ArgParser(default_config_files=['pointnetplus.cfg','pointnet.cfg', 'dgcnn.cfg'])
  parser.add('-c', '--cfg', required=True, is_config_file=True, help='config file path')
  parser.add('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
  parser.add('--dataset', type=str, default='touchstone3d', help='dataset name [touchstone, s3dis]')
  parser.add('--model', type=str, default='pointnet', help='model name [pointnet, pointnetplus, dgcnn, kpconv]')
  parser.add('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
  parser.add('--npoint', type=int, default=2048, help='Point Number [default: 2048]')
  # test
  parser.add('--checkpoint', type=str, default=None, help='Checkpoint to load for testing.')
  parser.add('--visual', type=bool, default=False, help='Create visualization for test data.')
  # debug
  parser.add('--debug', type=bool, default=False, help='allows to use log.debug')
  # log directory to save model and log
  parser.add('--log_dir', type=str, default=None, help='Log path [default: None]')
  
  return parser.parse_args()


