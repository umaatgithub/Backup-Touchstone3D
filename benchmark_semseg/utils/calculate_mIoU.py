import numpy as np
from sklearn.metrics import confusion_matrix

ep = 0.000001
#s_matrix = np.array([[1.0, 0.0, 0.0, 0.0],
#                     [0.0, 1.0, 1.0, 0.0],
#                     [0.0, 1.0, 1.0, 0.0],
#                     [0.0, 0.0, 0.0, 1.0]])

# false positive and false negative computation for SIOU
def fp_fn_sim(conf_matrix, s_matrix):
    return conf_matrix * s_matrix

# true positive computation for SIOU
def tp_sim(conf_matrix, s_matrix):
    # Multiply confusion matrix columns by s matrix columns
    column_products = np.sum(conf_matrix * s_matrix, axis=0)
    # Multiply confusion matrix rows by s matrix rows
    row_products = np.sum(conf_matrix * s_matrix, axis=1)
    # Sum the products to get the true positive
    tp = (column_products + row_products) / 2
    return tp

def calculate_siou(conf_matrix, s_matrix, d_matrix):
    # Similarity matrix IOU calculation
    tp = tp_sim(conf_matrix, s_matrix)
    fp_fn = fp_fn_sim(conf_matrix, d_matrix)
    fp = np.sum(fp_fn, axis=1)
    fn = np.sum(fp_fn, axis=0)
    siou = tp/(tp + fp + fn + ep)
    msiou = np.mean(siou)
    return siou, msiou

def metric_evaluate(predicted_label, gt_label, NUM_CLASS):
  """
  :param predicted_label: (B,N) tensor
  :param gt_label: (B,N) tensor
  :return: iou: scaler
  """
  
  s_matrix = np.eye(NUM_CLASS)
  d_matrix = 1 - s_matrix
  class_labels = [i for i in range(NUM_CLASS)]
  conf_matrix = np.zeros([NUM_CLASS, NUM_CLASS])
  gt_classes = [0 for _ in range(NUM_CLASS)]
  positive_classes = [0 for _ in range(NUM_CLASS)]
  true_positive_classes = [0 for _ in range(NUM_CLASS)]

  #for i in range(gt_label.size()[0]): @fayjie
  for i in range(gt_label.shape[0]):
    pred_pc = predicted_label[i]
    gt_pc = gt_label[i]
    #breakpoint()
    conf_matrix += confusion_matrix(gt_label.flatten(), predicted_label.flatten(), labels=class_labels)
    for j in range(gt_pc.shape[0]):
      gt_l = int(gt_pc[j])
      pred_l = int(pred_pc[j])
      gt_classes[gt_l] += 1
      positive_classes[pred_l] += 1
      true_positive_classes[gt_l] += int(gt_l == pred_l)

  siou, msiou = calculate_siou(conf_matrix, s_matrix, d_matrix)
  oa = sum(true_positive_classes)/float(sum(positive_classes))
  # print('Overall accuracy: {0}'.format(oa))
  
  iou_list = []
  for i in range(NUM_CLASS):
    iou_class = true_positive_classes[i] / float(gt_classes[i]+positive_classes[i]-true_positive_classes[i]+ep)
    iou_list.append(iou_class)

  mean_IoU = np.array(iou_list[:]).mean()

  return oa, mean_IoU, iou_list, msiou, siou
