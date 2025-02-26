import numpy as np

def metric_evaluate(predicted_label, gt_label, NUM_CLASS):
  """
  :param predicted_label: (B,N) tensor
  :param gt_label: (B,N) tensor
  :return: iou: scaler
  """
  gt_classes = [0 for _ in range(NUM_CLASS)]
  positive_classes = [0 for _ in range(NUM_CLASS)]
  true_positive_classes = [0 for _ in range(NUM_CLASS)]

  #for i in range(gt_label.size()[0]): @fayjie
  for i in range(gt_label.shape[0]):
    pred_pc = predicted_label[i]
    gt_pc = gt_label[i]

    for j in range(gt_pc.shape[0]):
      gt_l = int(gt_pc[j])
      pred_l = int(pred_pc[j])
      gt_classes[gt_l] += 1
      positive_classes[pred_l] += 1
      true_positive_classes[gt_l] += int(gt_l == pred_l)

  oa = sum(true_positive_classes)/float(sum(positive_classes))
  # print('Overall accuracy: {0}'.format(oa))
  
  iou_list = []
  for i in range(NUM_CLASS):
    iou_class = true_positive_classes[i] / float(gt_classes[i]+positive_classes[i]-true_positive_classes[i])
    iou_list.append(iou_class)

  mean_IoU = np.array(iou_list[1:]).mean()

  return oa, mean_IoU, iou_list
