import numpy as np
import matplotlib.pyplot as plt


def thresholdize(distID, distOOD, alpha = 0.5, plot = True):
  distID = distID[np.isfinite(distID)]
  distOOD = distOOD[np.isfinite(distOOD)]

  densID, binsID, _ = plt.hist(distID, bins = 100, density = True, alpha = 0.5, label = 'ID (Cats and Dogs)')
  densOOD, binsOOD, _ = plt.hist(distOOD, bins = 100, density = True, alpha = 0.5, label = 'OOD (Imgnet Animals & Textures)')
  if not plot:
    plt.clf()

  widthID = binsID[1]-binsID[0]
  widthOOD = binsOOD[1]-binsOOD[0]
  n = len(densID)

  low = np.min([binsID[0], binsOOD[0]])
  high = np.max([binsID[-1], binsOOD[-1]])
  thresholds = np.linspace(low, high, n)

  massID = np.zeros(n)
  massOOD = np.zeros(n)

  for i,x in enumerate(thresholds):
    massOOD[i] = np.sum(densOOD[binsOOD[:-1] < x]) * widthOOD
    massID[i] = np.sum(densID[binsID[:-1] >= x]) * widthID

  total = 2*(alpha*massOOD+ (1-alpha)*massID)
  thresIdx = np.argmax(total)
  threshold = thresholds[thresIdx]
  bestTotal = np.max(total) #not very intuitive when using alpha-thresholding

  if plot:
    plt.vlines(threshold, 0, 1.1*np.max(densOOD), label = 'Threshold', linestyles = 'dashed')
    plt.legend()

    plt.figure()
    plt.plot(thresholds, massID, label = 'ID mass right')
    plt.plot(thresholds, massOOD, label = 'OOD mass left')
    plt.plot(thresholds, total, label = 'Total')
    plt.vlines(threshold, 0, 1.1*np.max(total), label = 'Threshold', linestyles = 'dashed')
    plt.legend()

    falseOOD = 1.0 - massID[thresIdx]
    falseID = 1.0 - massOOD[thresIdx]

    print(f"Fraction of OOD data falsely classified as ID is {falseID:.3g}")
    print(f"Fraction of ID data falsely classified as OOD is {falseOOD:.3g}")

  return threshold


def thresholdize_freq(distID, distOOD, alpha = 0.5, plot = True):
  distID = distID[np.isfinite(distID)]
  distOOD = distOOD[np.isfinite(distOOD)]

  densID, binsID, _ = plt.hist(distID, bins = 100, density = False, alpha = 0.5, label = 'ID (Cats and Dogs)')
  densOOD, binsOOD, _ = plt.hist(distOOD, bins = 100, density = False, alpha = 0.5, label = 'OOD (Imgnet Animals & Textures)')
  if not plot:
    plt.clf()

  widthID = binsID[1]-binsID[0]
  widthOOD = binsOOD[1]-binsOOD[0]
  n = len(densID)

  low = np.min([binsID[0], binsOOD[0]])
  high = np.max([binsID[-1], binsOOD[-1]])
  thresholds = np.linspace(low, high, n)

  massID = np.zeros(n)
  massOOD = np.zeros(n)

  for i,x in enumerate(thresholds):
    massOOD[i] = np.sum(densOOD[binsOOD[:-1] < x]) * widthOOD
    massID[i] = np.sum(densID[binsID[:-1] >= x]) * widthID

  total = 2*(alpha*massOOD+ (1-alpha)*massID)
  thresIdx = np.argmax(total)
  threshold = thresholds[thresIdx]
  bestTotal = np.max(total) #not very intuitive when using alpha-thresholding

  if plot:
    plt.vlines(threshold, 0, 1.1*np.max(densOOD), label = 'Threshold', linestyles = 'dashed')
    plt.legend()

    plt.figure()
    plt.plot(thresholds, massID, label = 'ID mass right')
    plt.plot(thresholds, massOOD, label = 'OOD mass left')
    plt.plot(thresholds, total, label = 'Total')
    plt.vlines(threshold, 0, 1.1*np.max(total), label = 'Threshold', linestyles = 'dashed')
    plt.legend()

    falseOOD = 1.0 - massID[thresIdx]
    falseID = 1.0 - massOOD[thresIdx]

    print(f"Fraction of OOD data falsely classified as ID is {falseID:.3g}")
    print(f"Fraction of ID data falsely classified as OOD is {falseOOD:.3g}")

  return threshold


import sklearn.metrics as sk

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level, pos_label=1.):
    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]
   
    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true))), thresholds[cutoff]   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(in_examples, out_examples, recall_level=0.95):
    num_in = in_examples.shape[0]
    num_out = out_examples.shape[0]

    labels = np.zeros(num_in + num_out, dtype=np.int32)
    labels[:num_in] += 1

    examples = np.squeeze(np.vstack((in_examples, out_examples)))
    aupr_in = sk.average_precision_score(labels, examples)
    auroc = sk.roc_auc_score(labels, examples)

    fpr, cutoff_threshold = fpr_and_fdr_at_recall(labels, examples, recall_level)

    labels_rev = np.zeros(num_in + num_out, dtype=np.int32)
    labels_rev[num_in:] += 1
    examples = np.squeeze(-np.vstack((in_examples, out_examples)))
    aupr_out = sk.average_precision_score(labels_rev, examples)
    return auroc, aupr_in, aupr_out, fpr, cutoff_threshold
