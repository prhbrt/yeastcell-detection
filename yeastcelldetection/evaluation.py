# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import copy
from collections import Counter
import warnings


def calculate_metrics(results, *args):
    '''
    Calculate 4 standard performance metrics using performance indicators.
    Parameters
    ----------
    results : dict
        Contains at least 3 performance indicators: 
        true positives (tp), false positives (fp), false negatives (fn)
    Returns
    -------
    dict
        Performance metrics outcomes for 
        F1-score, accuracy, precision and recall.
    '''
    if len(args) > 0:
        warnings.warn('Using deprecated arguments in calculate_metrics, they will be removed.')
        
    precision = results["tp"]/(results["tp"]+results["fp"])
    recall = results["tp"]/(results["tp"]+results["fn"])
    accuracy = results["tp"]/(results["tp"]+results["fp"]+results["fn"])
    F = 2 * ((precision*recall) / (precision + recall))
    
    return {
        'F1-score': F, 'Accuracy': accuracy, 
        'Precision': precision, 'Recall': recall
    }


def match_detections_and_ground_truths(ground_truth, detections, masks):
  """Considering ground truth coordinates versus segmentation masks,
  yields tuples (ground_truth_index, detection_index) for every
  ground truth sample and detection in the same frame, such that the
  ground truth (x, y) coordinate matches the mask:
      # dataframe location, NOT index
      `masks[detection_location, y, x] == True`
  
  `detections` and `masks` must have the same length, as each item of mask is a
  height x width segmentation mask for that detection.
  Note that masks indices, should match detection locations.
  `ground_truths`, `detections` must be dataframes with columns
  [`frame`, `x`, `y`] and [`frame`, `mask`] respecitvely.
  
  Their indices must be unique.
  The `mask` column must point to the index of the mask for that detection,
  usually this is incremental from 0.
  """
  matches = []
  # iterate through grount truth and detected cells per time frame
  for frame, frame_ground_truths in ground_truth.groupby('frame'):
    frame_detections = detections[detections['frame'] == frame]
    frame_masks = masks[frame_detections['mask'].values]
    x, y = np.round(frame_ground_truths[['x', 'y']].values).astype(int).T
    mask_values_at_yx = frame_masks[:, y, x]
    found = mask_values_at_yx.sum(0) > 0
    detection_indices, ground_truth_indices = np.where(mask_values_at_yx)
    found = found[ground_truth_indices]
    matches.extend(zip(
        frame_ground_truths.index[ground_truth_indices[found]],
        frame_detections.index[detection_indices[found]]))
  return pd.DataFrame(matches, columns=['ground truth index', 'detection index'])


def get_segmention_metrics(ground_truth, detections, masks):
  """For the segmentation task, returns how many true positives and true/false
  positives as a dictionary including how many ground truths were detected by
  the same mask (merged).
  
  Arguments the same as `match_detections_and_ground_truths`"""
  matches = match_detections_and_ground_truths(ground_truth, detections, masks)

  # We defined true positives to be those ground truths that are picked up as
  # a detection, but only if this detection has only one ground truths
  # assinged to it.

  # figuring out which detections have more than one ground truth, they 'join'
  # ground truths together, and filtering out matches on this detection.
  detection_joining_gt = matches.groupby('detection index').count() > 1
  detection_joining_gt = detection_joining_gt.index[detection_joining_gt['ground truth index']]
  unjoined_matches = matches[~matches['detection index'].isin(detection_joining_gt)]
  # then the amount of true positives, equals the amount of ground truths that
  # still have a detection assigned.
  tp = len(unjoined_matches['ground truth index'].unique())

  # Anything that
  split = int((unjoined_matches.groupby('ground truth index').count() - 1).sum())
  metrics = {
    'tp': tp,
    'fp': len(set(detections.index) - set(unjoined_matches['detection index'])) + split,
    'fn': len(set(ground_truth.index) - set(unjoined_matches['ground truth index'])),
    'join': len(detection_joining_gt), 'split': split,
  }
  return metrics


def compare_links(a, b, mapping):
  a = a[a['cell'] >= 0].copy()
  a['other index'] = [mapping.get(i, -2) for i in a.index]
  a['previous frame'] = a['frame'] - 1

  to_other = pd.merge(
    a, a, how='inner',
    left_on=['frame','cell'],
    right_on=['previous frame', 'cell']
  )[['other index_x', 'other index_y']].applymap(
      lambda value: (value if value < 0 else int(b.loc[value]['cell'])))
  to_other.columns = [('other cell', 't'), ('other cell', 't+1')]

  propagated = (to_other < 0).max(1)
  to_other = to_other[~propagated]
  outliers = (to_other == -1).max(1)
  to_other = to_other[~outliers]

  true = (to_other[('other cell', 't')] == to_other[('other cell', 't+1')]).sum()
  false = (to_other[('other cell', 't')] != to_other[('other cell', 't+1')]).sum()

  return {'untracked': outliers.sum(), 'unmapped': propagated.sum(),
          'true': true, 'false': false + outliers.sum() + propagated.sum()}


def get_tracking_metrics(ground_truth, detections, masks):
  # If multiple detections got the same label in a frame,
  # either select the one with the highest segmentation score or
  # the first one.
  if 'segmentation_score' in detections.columns:
    sorted_detections = detections.sort_values(
      ['frame', 'cell', 'segmentation_score'], ascending=False)

    best_detection_in_frame = (
        (sorted_detections.groupby(['frame', 'cell']).cumcount() == 0) |
        (sorted_detections['cell'] < 0))

    det = detections.loc[best_detection_in_frame]
    overmatching = (~best_detection_in_frame).sum()
  else:
    first_detection_in_frame = (
      (detections.groupby(['frame', 'cell']).cumcount() == 0) |
      (detections['cell'] < 0))
    det = detections.loc[first_detection_in_frame]
    overmatching = (~first_detection_in_frame).sum()

  matches = match_detections_and_ground_truths(ground_truth, det, masks)
  detection_joining_gt = matches.groupby('detection index')['ground truth index'].count() > 1
  detection_joining_gt = detection_joining_gt.index[detection_joining_gt]
  unjoined_matches = matches[~matches['detection index'].isin(detection_joining_gt)]

  assert (unjoined_matches.groupby('detection index')['ground truth index']
          .count() > 1).sum() == 0,(
    "Uncanny, joins should have been removed")

  det_to_gt = unjoined_matches.groupby('detection index')['ground truth index'].first()
  gt_to_det = unjoined_matches.groupby('ground truth index')['detection index'].first()
  comparison_gt = compare_links(ground_truth, det, gt_to_det)
  comparison_det = compare_links(det, ground_truth, det_to_gt)
  assert comparison_det['true'] == comparison_det['true'], (
      "Uncanny, different links matches going from ground truth to "
      "detections as vice versa. This shouldn't happen")
  assert comparison_det['untracked'] == 0, (
      "Uncanny, ground truth should not have untracked (cell == -1) cell labels")
  return {'tp': comparison_gt['true'], 'fp': comparison_det['false'],
          'fn': comparison_gt['false'],
          # when a cell was tracked multiple times in a frame.
          'over matching': overmatching,
          # ground truth links matched in segmentation but note tracked.
          'tracking fn': comparison_gt['untracked'],
          # also specify propagated segmentation errors
          'segmentation fn': comparison_gt['unmapped'],
          'segmentation fp': comparison_det['unmapped'],
          'untracked fn': comparison_det['untracked'],}
