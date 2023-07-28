import copy
import numpy as np
from detectron2.utils import base_storage
from scipy.optimize import linear_sum_assignment


def eval_nucleus(result, coco, coco_logger, class_id=None):
    import pycocotools.mask as maskutil
    result = copy.deepcopy(result)
    if class_id is not None:
        result = [x for x in result if x['score'] >= 0.5 and x['category_id'] == class_id]
    else:
        result = [x for x in result if x['score'] >= 0.5]
    result = sorted(result, key=lambda r: r['image_id'])

    ids = list(sorted(coco.imgs.keys()))
    img_id_len = max(ids)

    all_height = []
    all_width = []
    maskall = ([])
    for ii in range(1, img_id_len + 1):
        mask0 = None
        img_height = img_width = None
        for i, c in enumerate(result):
            if c['image_id'] == ii:
                if mask0 is None:
                    img_height, img_width = result[i]['segmentation']['size']
                    mask0 = np.zeros((img_height, img_width), dtype=int)
                mask = maskutil.decode(result[i]['segmentation']).astype('int')
                mask = np.reshape(mask, (img_height, img_width))
                mask[mask > 0] = i + 1
                mask0[mask0 == 0] = mask[mask0 == 0]
            elif mask0 is not None:
                result = result[i:]
                break
        all_height.append(img_height)
        all_width.append(img_width)
        mask0 = remap_label(mask0)
        maskall.append(mask0)

    gtall = []
    for ii in range(1, img_id_len + 1):
        gt0 = None
        img_height, img_width = all_height[ii - 1], all_width[ii - 1]
        if img_height is not None:
            if class_id is not None:
                anno_ids = coco.getAnnIds(imgIds=ii, catIds=class_id, iscrowd=None)
            else:
                anno_ids = coco.getAnnIds(imgIds=ii, catIds=[], iscrowd=None)
            annotations = coco.loadAnns(anno_ids)
            for i, c in enumerate(annotations):
                if gt0 is None:
                    gt0 = np.zeros((img_height, img_width), dtype=int)
                each_gt = coco.annToMask(c).astype('int')
                each_gt = np.reshape(each_gt, (img_height, img_width))
                each_gt[each_gt > 0] = i + 1
                gt0[gt0 == 0] = each_gt[gt0 == 0]
            gt0 = remap_label(gt0)
            gtall.append(gt0)
        else:
            gtall.append([1])

    dice_score = []
    AJI_score = []
    dq_score = []
    sq_score = []
    pq_score = []
    for i in range(img_id_len):
        if np.max(gtall[i]) == 0 and np.max(maskall[i]) == 0:
            dice_score.append(1)
            AJI_score.append(1)
            dq_score.append(1)
            sq_score.append(1)
            pq_score.append(1)
            continue
        elif np.max(gtall[i]) == 0:
            dice_score.append(0)
            AJI_score.append(0)
            dq_score.append(0)
            sq_score.append(0)
            pq_score.append(0)
            continue
        elif np.max(maskall[i]) == 0:
            dice_score.append(0)
            AJI_score.append(0)
            dq_score.append(0)
            sq_score.append(0)
            pq_score.append(0)
            continue
        else:
            dice = dice_coefficient(true=gtall[i], pred=maskall[i])
            aji = get_fast_aji(true=gtall[i], pred=maskall[i])
            pq_info = get_fast_pq(true=gtall[i], pred=maskall[i])[0]
            dice_score.append(dice)
            AJI_score.append(aji)
            dq_score.append(pq_info[0])
            sq_score.append(pq_info[1])
            pq_score.append(pq_info[2])

    bad_dice, bad_id = get_extreme_sample(dice_score, 10, 'bad')
    base_storage.set_value('bad_id', bad_id)
    bad_name = []
    for id in bad_id:
        bad_name.append(coco.loadImgs(id)[0]['file_name'])
    coco_logger.info(f'10 worst example id: {bad_id}')
    coco_logger.info(f'10 worst img_name: {bad_name}')
    coco_logger.info(f'corresponding dice: {bad_dice}')
    good_dice, good_id = get_extreme_sample(dice_score, 10, 'good')
    good_name = []
    for id in good_id:
        good_name.append(coco.loadImgs(id)[0]['file_name'])
    base_storage.set_value('good_id', good_id)
    coco_logger.info(f'10 best example id: {good_id}')
    coco_logger.info(f'10 best img_name: {good_name}')
    coco_logger.info(f'corresponding dice: {good_dice}')
    Dice = np.array(dice_score).mean()
    AJI = np.array(AJI_score).mean()
    DQ = np.array(dq_score).mean()
    SQ = np.array(sq_score).mean()
    PQ = np.array(pq_score).mean()

    return Dice, AJI, DQ, SQ, PQ, bad_id, good_id


def eval_multiclass_nucleus(result, coco, nr_classes=6):
    import pycocotools.mask as maskutil
    result = copy.deepcopy(result)
    result = [x for x in result if x['score'] >= 0.5]
    result = sorted(result, key=lambda r: r['image_id'])

    ids = list(sorted(coco.imgs.keys()))
    img_id_len = max(ids)

    all_height = []
    all_width = []
    mask_all = []
    mask_bin_all = []

    for ii in range(1, img_id_len + 1):
        mask0 = None
        mask1 = None
        img_height = img_width = 256
        for i, c in enumerate(result):
            if c['image_id'] == ii:
                if mask0 is None:
                    img_height, img_width = result[i]['segmentation']['size']
                    mask0 = np.zeros((img_height, img_width), dtype=int)
                    mask1 = np.zeros((img_height, img_width), dtype=int)
                mask_bin = maskutil.decode(result[i]['segmentation'])
                mask_bin = np.reshape(mask_bin, (img_height, img_width))
                mask = mask_bin.copy().astype('bool').astype('int')
                mask *= i + 1
                zero_part = mask0 == 0
                mask0[zero_part] = mask[zero_part]
                mask1[zero_part] = (mask_bin[zero_part]).astype('int') * int(c['category_id'])
            elif mask0 is not None:
                result = result[i:]
                break
        all_height.append(img_height)
        all_width.append(img_width)
        if mask0 is None:
            cat_mask = np.zeros((img_height, img_width, 2), dtype=int)
            mask0 = np.zeros((img_height, img_width), dtype=int)
        else:
            cat_mask = np.concatenate((mask0[:, :, np.newaxis], mask1[:, :, np.newaxis]), axis=-1)
        mask_all.append(cat_mask)
        mask_bin_all.append(remap_label(mask0))

    gtall = base_storage.get_value('gt_all')
    gtbin = base_storage.get_value('gt_bin')
    if gtall is None:
        gtall = []
        gtbin = []
        for ii in range(1, img_id_len + 1):
            gt0 = None
            gt1 = None
            img_height, img_width = all_height[ii - 1], all_width[ii - 1]
            anno_ids = coco.getAnnIds(imgIds=ii, iscrowd=None)
            annotations = coco.loadAnns(anno_ids)
            for i, c in enumerate(annotations):
                if gt0 is None:
                    gt0 = np.zeros((img_height, img_width), dtype=int)
                    gt1 = np.zeros((img_height, img_width), dtype=int)
                each_gt_bin = coco.annToMask(c)
                each_gt_bin = np.reshape(each_gt_bin, (img_height, img_width))
                each_gt = each_gt_bin.copy().astype('bool').astype('int')
                each_gt[each_gt > 0] = i + 1
                zero_part = gt0 == 0
                gt0[zero_part] = each_gt[zero_part]
                gt1[zero_part] = (each_gt_bin[zero_part]).astype('int') * int(c['category_id'])
            if gt0 is None:
                gt_mask = np.zeros((img_height, img_width, 2), dtype=int)
                gt0 = np.zeros((img_height, img_width), dtype=int)
            else:
                gt_mask = np.concatenate((gt0[:, :, np.newaxis], gt1[:, :, np.newaxis]), axis=-1)
            gtall.append(gt_mask)
            gtbin.append(remap_label(gt0))
        base_storage.set_value('gt_all', gtall)
        base_storage.set_value('gt_bin', gtbin)

    dices = []
    AJIs = []
    for i in range(img_id_len):
        dices.append(dice_coefficient(true=gtbin[i], pred=mask_bin_all[i]))
        AJIs.append(get_fast_aji(true=gtbin[i], pred=mask_bin_all[i]))
    dice_score = np.mean(dices)
    AJI = np.mean(AJIs)

    mpq_info_list = []
    bpq_info_list = []
    for patch_idx in range(len(gtall)):
        mpq_info_single = get_multi_pq_info(gtall[patch_idx], mask_all[patch_idx],  nr_classes=nr_classes)
        pq_oneclass_info = get_pq(gtall[patch_idx][..., 0], mask_all[patch_idx][..., 0])
        mpq_info = []
        # aggregate the stat info per class
        for single_class_pq in mpq_info_single:
            tp = single_class_pq[0]
            fp = single_class_pq[1]
            fn = single_class_pq[2]
            sum_iou = single_class_pq[3]
            mpq_info.append([tp, fp, fn, sum_iou])
        mpq_info_list.append(mpq_info)

        bpq_info = [
            pq_oneclass_info[1][0],
            pq_oneclass_info[1][1],
            pq_oneclass_info[1][2],
            pq_oneclass_info[2],
        ]
        bpq_info_list.append(list(bpq_info))

    mpq_info_metrics = np.array(mpq_info_list, dtype="float")
    bpq_info_metrics = np.array(bpq_info_list, dtype="float")
    # sum over all the images
    total_mpq_info_metrics = np.sum(mpq_info_metrics, axis=0)
    total_bpq_info_metrics = np.sum(bpq_info_metrics, axis=0)
    mdq_list = []
    msq_list = []
    mpq_list = []
    # for each class, get the multiclass PQ
    for cat_idx in range(total_mpq_info_metrics.shape[0]):
        total_tp = total_mpq_info_metrics[cat_idx][0]
        total_fp = total_mpq_info_metrics[cat_idx][1]
        total_fn = total_mpq_info_metrics[cat_idx][2]
        total_sum_iou = total_mpq_info_metrics[cat_idx][3]

        # get the F1-score i.e DQ
        dq = total_tp / (
                (total_tp + 0.5 * total_fp + 0.5 * total_fn) + 1.0e-6
        )
        # get the SQ, when not paired, it has 0 IoU so does not impact
        sq = total_sum_iou / (total_tp + 1.0e-6)
        mdq_list.append(dq)
        msq_list.append(sq)
        mpq_list.append(dq * sq)
    mdq_metrics = np.array(mdq_list)
    msq_metrics = np.array(msq_list)
    mpq_metrics = np.array(mpq_list)
    mdq = np.mean(mdq_metrics)
    msq = np.mean(msq_metrics)
    mpq = np.mean(mpq_metrics)

    total_tp = total_bpq_info_metrics[0]
    total_fp = total_bpq_info_metrics[1]
    total_fn = total_bpq_info_metrics[2]
    total_sum_iou = total_bpq_info_metrics[3]

    # get the F1-score i.e DQ
    bdq = total_tp / (
            (total_tp + 0.5 * total_fp + 0.5 * total_fn) + 1.0e-6
    )
    # get the SQ, when not paired, it has 0 IoU so does not impact
    bsq = total_sum_iou / (total_tp + 1.0e-6)
    bpq = bdq * bsq

    return dice_score, AJI, bpq, mdq, msq, mpq


def get_extreme_sample(metric_list, num, mode='bad'):
    t = copy.deepcopy(metric_list)
    # 求m个最大的数值及其索引
    extreme_number = []
    extreme_index = []
    for _ in range(num):
        if mode == 'bad':
            number = min(t)
            index = t.index(number)
            t[index] = 1
        elif mode == 'good':
            number = max(t)
            index = t.index(number)
            t[index] = 0
        extreme_number.append(number)
        extreme_index.append(index + 1)
    return extreme_number, extreme_index


def dice_coefficient(true, pred):
    smooth = 1
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    inter = true * pred
    denom = true + pred
    return 2.0 * (np.sum(inter) + smooth) / (np.sum(denom) + smooth)


def dice_coefficient_2(true, pred):
    true = np.copy(true)
    pred = np.copy(pred)
    true_id = list(np.unique(true))
    pred_id = list(np.unique(pred))
    # remove background aka id 0
    true_id.remove(0)
    pred_id.remove(0)

    if len(true_id):
        total_markup = 0
        total_intersect = 0
        for t in true_id:
            t_mask = np.array(true == t, np.uint8)
            for p in pred_id:
                p_mask = np.array(pred == p, np.uint8)
                intersect = p_mask * t_mask
                if intersect.sum() > 0:
                    total_intersect += intersect.sum()
                    total_markup += t_mask.sum() + p_mask.sum()
        return 2 * total_intersect / total_markup
    else:
        return 0


def get_fast_aji(true, pred):
    """AJI version distributed by MoNuSeg, has no permutation problem but suffered from
    over-penalisation similar to DICE2.
    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no
    effect on the result.
    """
    true = np.copy(true)  # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))
    if len(true_id_list) == 1 and len(pred_id_list) == 1:
        return 1
    elif len(true_id_list) == 1 or len(pred_id_list) == 1:
        return 0

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, dont care
    # about reusing pred instance multiple times
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()

    paired_true = list(paired_true + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    aji_score = overall_inter / overall_union
    return aji_score


def AJI(p, g):
    p_ind = np.ones(p.shape[2])
    g_ind = np.ones(g.shape[2])
    I = 0
    U = 0
    for i in range(g.shape[2]):
        iou0 = 0
        ind = -1
        for j in range(p.shape[2]):
            iou = (g[..., i] * p[..., j]).sum() / (g[..., i] + p[..., j]).astype(bool).sum()
            if iou > iou0:
                iou0 = iou
                ind = j
        if ind != -1:
            p_ind[ind] = 0
            g_ind[i] = 0

            I = I + (g[..., i] * p[..., ind]).sum()

            U = U + (g[..., i] + p[..., ind]).astype(bool).sum()
            p[..., ind] = 0
        # else:
        #    I = I+(g[...,i]*p[...,0]).sum()

        #    U = U+(g[...,i]+p[...,0]).astype(bool).sum()

    U = U + p.sum() + g[..., g_ind.astype(bool)].sum()
    return I / U, I, U


def get_fast_pq(true, pred, match_iou=0.5):
    """`match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique
    (1 prediction instance to 1 GT instance mapping).
    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing.
    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.

    Fast computation requires instance IDs are in contiguous orderding
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand
    and `by_size` flag has no effect on the result.
    Returns:
        [dq, sq, pq]: measurement statistic
        [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                      pairing information to perform measurement

    """
    assert match_iou >= 0.0, "Cant' be negative"

    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]


def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.
    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    if pred is None:
        return [0]
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


def get_bounding_box(img):
    """Get the bounding box coordinates of a binary input- assumes a single object.
    Args:
        img: input binary image.
    Returns:
        bounding box coordinates
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def get_multi_pq_info(true, pred, nr_classes=6, match_iou=0.5):
    """Get the statistical information needed to compute multi-class PQ.

    CoNIC multiclass PQ is achieved by considering nuclei over all images at the same time,
    rather than averaging image-level results, like was done in MoNuSAC. This overcomes issues
    when a nuclear category is not present in a particular image.

    Args:
        true (ndarray): HxWx2 array. First channel is the instance segmentation map
            and the second channel is the classification map.
        pred: HxWx2 array. First channel is the instance segmentation map
            and the second channel iget_multi_pq_infos the classification map.
        nr_classes (int): Number of classes considered in the dataset.
        match_iou (float): IoU threshold for determining whether there is a detection.

    Returns:
        statistical info per class needed to compute PQ.

    """
    assert match_iou >= 0.0, "Cant' be negative"

    true_inst = true[..., 0]
    pred_inst = pred[..., 0]
    ###
    true_class = true[..., 1]
    pred_class = pred[..., 1]

    pq = []
    for idx in range(nr_classes):
        pred_class_tmp = pred_class == idx + 1
        pred_inst_oneclass = pred_inst * pred_class_tmp
        pred_inst_oneclass = remap_label(pred_inst_oneclass)
        ##
        true_class_tmp = true_class == idx + 1
        true_inst_oneclass = true_inst * true_class_tmp
        true_inst_oneclass = remap_label(true_inst_oneclass)

        pq_oneclass_info = get_pq(true_inst_oneclass, pred_inst_oneclass, remap=False)

        # add (in this order) tp, fp, fn iou_sum
        pq_oneclass_stats = [
            pq_oneclass_info[1][0],
            pq_oneclass_info[1][1],
            pq_oneclass_info[1][2],
            pq_oneclass_info[2],
        ]
        pq.append(pq_oneclass_stats)

    return pq


def get_pq(true, pred, match_iou=0.5, remap=True):
    """Get the panoptic quality result.

    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `remap_label` beforehand. Here, the `by_size` flag
    has no effect on the result.
    Args:
        true (ndarray): HxW ground truth instance segmentation map
        pred (ndarray): HxW predicted instance segmentation map
        match_iou (float): IoU threshold level to determine the pairing between
            GT instances `p` and prediction instances `g`. `p` and `g` is a pair
            if IoU > `match_iou`. However, pair of `p` and `g` must be unique
            (1 prediction instance to 1 GT instance mapping). If `match_iou` < 0.5,
            Munkres assignment (solving minimum weight matching in bipartite graphs)
            is caculated to find the maximal amount of unique pairing. If
            `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
            the number of pairs is also maximal.
        remap (bool): whether to ensure contiguous ordering of instances.

    Returns:
        [dq, sq, pq]: measurement statistic
        [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                      pairing information to perform measurement

        paired_iou.sum(): sum of IoU within true positive predictions

    """
    assert match_iou >= 0.0, "Cant' be negative"
    # ensure instance maps are contiguous
    if remap:
        pred = remap_label(pred)
        true = remap_label(true)

    true = np.copy(true)
    pred = np.copy(pred)
    true = true.astype("int32")
    pred = pred.astype("int32")
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))
    # prefill with value
    pairwise_iou = np.zeros([len(true_id_list), len(pred_id_list)], dtype=np.float64)

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask_lab = true == true_id
        rmin1, rmax1, cmin1, cmax1 = get_bounding_box(t_mask_lab)
        t_mask_crop = t_mask_lab[rmin1:rmax1, cmin1:cmax1]
        t_mask_crop = t_mask_crop.astype("int")
        p_mask_crop = pred[rmin1:rmax1, cmin1:cmax1]
        pred_true_overlap = p_mask_crop[t_mask_crop > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask_lab = pred == pred_id
            p_mask_lab = p_mask_lab.astype("int")

            # crop region to speed up computation
            rmin2, rmax2, cmin2, cmax2 = get_bounding_box(p_mask_lab)
            rmin = min(rmin1, rmin2)
            rmax = max(rmax1, rmax2)
            cmin = min(cmin1, cmin2)
            cmax = max(cmax1, cmax2)
            t_mask_crop2 = t_mask_lab[rmin:rmax, cmin:cmax]
            p_mask_crop2 = p_mask_lab[rmin:rmax, cmin:cmax]

            total = (t_mask_crop2 + p_mask_crop2).sum()
            inter = (t_mask_crop2 * p_mask_crop2).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou

    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / ((tp + 0.5 * fp + 0.5 * fn) + 1.0e-6)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return (
        [dq, sq, dq * sq],
        [tp, fp, fn],
        paired_iou.sum(),
    )
