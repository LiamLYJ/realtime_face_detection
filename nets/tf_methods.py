# convert numpy methods to tensorflow methods
# for adding post-processing on graph
import tensorflow as tf
import sys
sys.path.append("..")
import tf_extended as tfe

def tf_ssd_bboxes_select(predictions_net,
                         localizations_net,
                         anchors_net,
                         select_threshold=0.5,
                         img_shape = (300,300),
                         num_classes=21,
                         decode = True,
                         ignore_class=0,
                         scope=None):
    """Extract classes, scores and bounding boxes from network output layers.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_net: List of SSD prediction layers;
      localizations_net: List of localization layers;
      select_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
    """
    with tf.name_scope(scope, 'ssd_bboxes_select',
                       [predictions_net, localizations_net]):
        l_classes = []
        l_scores = []
        l_bboxes = []
        # print ('len of predictions:',len(predictions_net))
        # print ('shape of one predictions', predictions_net[0].shape)
        for i in range(len(predictions_net)):
            classes, scores, bboxes = tf_ssd_bboxes_select_layer(predictions_net[i],
                                                        localizations_net[i],
                                                        anchors_net[i],
                                                        select_threshold,
                                                        img_shape,
                                                        num_classes,
                                                        decode)
            l_classes.append(classes)
            l_scores.append(scores)
            l_bboxes.append(bboxes)
        # Concat results.
        # print ('shape of l_classes:',l_classes[0].shape)
        # print ('shape of l_scores:',l_scores[0].shape)
        # print ('shape of l_bboxes:',l_bboxes[0].shape)
        classes = tf.concat(l_classes,0)
        scores = tf.concat(l_scores,0)
        bboxes = tf.concat(l_bboxes,0)
        # print ('shape of classes:',classes.shape)
        # print ('shape of scores:',scores.shape)
        # print ('shape of bboxes:',bboxes.shape)
        # raise
        return classes, scores, bboxes


def tf_ssd_bboxes_select_layer(predictions_layer,
                               localizations_layer,
                               anchors_layer,
                               select_threshold=0.5,
                               img_shape = (300,300),
                               num_classes=21,
                               decode = True,
                               ignore_class=0,
                               scope=None):
    """Extract classes, scores and bounding boxes from features in one layer.
    Batch-compatible: inputs are supposed to have batch-type shapes.

    Args:
      predictions_layer: A SSD prediction layer;
      localizations_layer: A SSD localization layer;
      select_threshold: Classification threshold for selecting a box. All boxes
        under the threshold are set to 'zero'. If None, no threshold applied.
    Return:
      d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
        size Batches X N x 1 | 4. Each key corresponding to a class.
    """
    if decode:
        localizations_layer = tf_ssd_bboxes_decode(localizations_layer, anchors_layer)
    with tf.name_scope(scope, 'ssd_bboxes_select_layer',
                       [predictions_layer, localizations_layer]):
        # Reshape features: Batches x N x N_labels | 4
        # print ('predictions_layer shape:',predictions_layer.shape)
        p_shape = tfe.get_shape(predictions_layer)
        predictions_layer = tf.reshape(predictions_layer,
                                       tf.stack([p_shape[0], -1, p_shape[-1]]))
        l_shape = tfe.get_shape(localizations_layer)
        localizations_layer = tf.reshape(localizations_layer,
                                         tf.stack([l_shape[0], -1, l_shape[-1]]))
        # Remove boxes under the threshold.
        sub_predictions = predictions_layer[:, :, 1:]

        _sub_predictions = tf.squeeze(sub_predictions)
        idxes = tf.where(_sub_predictions > select_threshold)
        sub_predictions = tf.squeeze(sub_predictions,axis = 0)

        localizations_layer = tf.squeeze(localizations_layer)

        # can't use tf.gather_nd,it's not registed on IOS-tensorflow
        # idxse = tf.cast(idxes,tf.int32)
        # bboxes = tf.gather_nd(localizations_layer,idxes)
        # scores = tf.gather_nd(sub_predictions,idxes)

        # use tf.gather method
        # idxes = tf.squeeze(tf.cast(idxes, tf.int32))
        idxse = tf.cast(idxes,tf.int32)
        scores = tf.squeeze(tf.gather(sub_predictions,idxes),axis = 1)
        bboxes = tf.squeeze(tf.gather(localizations_layer,idxes), axis =1)

        # print ('shape of idxes:',idxes.shape)
        # print ('shape of sub_predictions:',sub_predictions.shape)
        # print ('shape of localizations_net:',localizations_layer.shape)
        # print ('shape of bboxes:',bboxes.shape)
        # print ('shape of scores:',scores.shape)
        # raise

        classes = tf.ones_like(scores)
        # print ('shape of classes:',classes.shape)
        # print ('shape of scores:',scores.shape)
        # print ('shape of bboxes:',bboxes.shape)
        return classes, scores, bboxes


def tf_ssd_bboxes_decode(feat_localizations,
                               anchors_layer,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2]):
    """Compute the relative bounding boxes from the layer features and
    reference anchor bounding boxes.

    Arguments:
      feat_localizations: Tensor containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      Tensor Nx4: ymin, xmin, ymax, xmax
    """

    yref, xref, href, wref = anchors_layer

    # Compute center, height and width
    cx = feat_localizations[:, :, :, :, 0] * wref * prior_scaling[0] + xref
    cy = feat_localizations[:, :, :, :, 1] * href * prior_scaling[1] + yref
    w = wref * tf.exp(feat_localizations[:, :, :, :, 2] * prior_scaling[2])
    h = href * tf.exp(feat_localizations[:, :, :, :, 3] * prior_scaling[3])
    # Boxes coordinates.
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    return bboxes

def tf_bboxes_clip(bbox_ref, bboxes):
    return tfe.bboxes_clip(bbox_ref, bboxes)

def tf_bboxes_sort(classes, scores, bboxes, top_k = 400):
    return tfe.bboxes_sort_all_classes(classes, scores, bboxes,top_k)

def tf_bboxes_resize(bbox_ref, bboxes):
    return tfe.bboxes_resize(bbox_ref, bboxes)

def tf_bboxes_nms(classes,scores,bboxes, nms_threshold = 0.45):
    # scores, bboxes = tfe.bboxes_nms_batch(scores, bboxes, nms_threshold)
    scores, bboxes = tfe.bboxes_nms(scores, bboxes, nms_threshold)
    classes = tf.ones_like(scores)
    return classes, scores, bboxes
