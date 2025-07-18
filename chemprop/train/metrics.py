from typing import List, Callable, Union

from tqdm import trange
import torch
import numpy as np
import torch.nn as nn

from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score, \
    roc_auc_score, accuracy_score, log_loss, f1_score, matthews_corrcoef, recall_score, precision_score, \
    balanced_accuracy_score


def get_metric_func(metric: str) -> Callable[[Union[List[int], List[float]], List[float]], float]:
    r"""
    Gets the metric function corresponding to a given metric name.

    Supports:

    * :code:`auc`: Area under the receiver operating characteristic curve
    * :code:`prc-auc`: Area under the precision recall curve
    * :code:`rmse`: Root mean squared error
    * :code:`mse`: Mean squared error
    * :code:`mae`: Mean absolute error
    * :code:`r2`: Coefficient of determination R\ :superscript:`2`
    * :code:`accuracy`: Accuracy (using a threshold to binarize predictions)
    * :code:`cross_entropy`: Cross entropy
    * :code:`binary_cross_entropy`: Binary cross entropy
    * :code:`sid`: Spectral information divergence
    * :code:`wasserstein`: Wasserstein loss for spectra
    * :code:`balanced_accuracy`: Balanced accuracy
    * :code:`recall`: Recall
    * :code:`precision`: Precision

    :param metric: Metric name.
    :return: A metric function which takes as arguments a list of targets and a list of predictions and returns.
    """
    if metric == 'auc':
        return roc_auc_score

    if metric == 'prc-auc':
        return prc_auc

    if metric == 'rmse':
        return rmse

    if metric == 'mse':
        return mean_squared_error

    if metric == 'mae':
        return mean_absolute_error

    if metric == 'bounded_rmse':
        return bounded_rmse

    if metric == 'bounded_mse':
        return bounded_mse

    if metric == 'bounded_mae':
        return bounded_mae

    if metric == 'r2':
        return r2_score

    if metric == 'accuracy':
        return accuracy

    if metric == 'cross_entropy':
        return log_loss

    if metric == 'f1':
        return f1_metric

    if metric == 'mcc':
        return mcc_metric

    if metric == 'binary_cross_entropy':
        return bce

    if metric == 'sid':
        return sid_metric

    if metric == 'wasserstein':
        return wasserstein_metric

    if metric == 'balanced_accuracy':
        return balanced_accuracy_metric

    if metric == 'recall':
        return recall_metric

    if metric == 'precision':
        return precision_metric
    
    # if metric == 'squared_log_fugacity_difference':
    #     return squared_log_fugacity_difference_metric

    raise ValueError(f'Metric "{metric}" not supported.')


def compute_hard_predictions(preds, threshold=0.5):
    """
    Compute hard predictions from model outputs.

    Args:
    - preds (list): A list of predictions, either probabilities for binary classification or class probabilities for multiclass classification.
    - threshold (float, optional): Threshold for converting probabilities to binary outcomes in binary classification. Defaults to 0.5.

    Returns:
    - hard_preds (list): A list of hard predictions (0/1 for binary, class index for multiclass).
    """
    if preds and isinstance(preds[0], list):  # Multiclass prediction
        return [p.index(max(p)) for p in preds]
    else:  # Binary prediction
        return [1 if p > threshold else 0 for p in preds]


def prc_auc(targets: List[int], preds: List[float]) -> float:
    """
    Computes the area under the precision-recall curve.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed prc-auc.
    """
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)


def bce(targets: List[int], preds: List[float]) -> float:
    """
    Computes the binary cross entropy loss.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed binary cross entropy.
    """
    # Don't use logits because the sigmoid is added in all places except training itself
    bce_func = nn.BCELoss(reduction='mean')
    loss = bce_func(target=torch.Tensor(targets), input=torch.Tensor(preds)).item()

    return loss


def rmse(targets: List[float], preds: List[float]) -> float:
    """
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    return mean_squared_error(targets, preds, squared=False)


def bounded_rmse(targets: List[float], preds: List[float], gt_targets: List[bool] = None,
                 lt_targets: List[bool] = None) -> float:
    """
    Computes the root mean squared error, considering targets with inequalities.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :param gt_targets: A list of booleans indicating whether the target is a >target inequality.
    :param lt_targets: A list of booleans indicating whether the target is a <target inequality.
    :return: The computed rmse.
    """
    # When the target is a greater-than-inequality and the prediction is greater than the target,
    # replace the prediction with the target. Analogous for less-than-inequalities.
    preds = np.where(
        np.logical_and(np.greater(preds, targets), gt_targets),
        targets,
        preds,
    )
    preds = np.where(
        np.logical_and(np.less(preds, targets), lt_targets),
        targets,
        preds,
    )
    return mean_squared_error(targets, preds, squared=False)


def bounded_mse(targets: List[float], preds: List[float], gt_targets: List[bool] = None,
                lt_targets: List[bool] = None) -> float:
    """
    Computes the mean squared error, considering targets with inequalities.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :param gt_targets: A list of booleans indicating whether the target is a >target inequality.
    :param lt_targets: A list of booleans indicating whether the target is a <target inequality.
    :return: The computed mse.
    """
    # When the target is a greater-than-inequality and the prediction is greater than the target,
    # replace the prediction with the target. Analogous for less-than-inequalities.
    preds = np.where(
        np.logical_and(np.greater(preds, targets), gt_targets),
        targets,
        preds,
    )
    preds = np.where(
        np.logical_and(np.less(preds, targets), lt_targets),
        targets,
        preds,
    )
    return mean_squared_error(targets, preds, squared=True)


def bounded_mae(targets: List[float], preds: List[float], gt_targets: List[bool] = None,
                lt_targets: List[bool] = None) -> float:
    """
    Computes the mean absolute error, considering targets with inequalities.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :param gt_targets: A list of booleans indicating whether the target is a >target inequality.
    :param lt_targets: A list of booleans indicating whether the target is a <target inequality.
    :return: The computed mse.
    """
    # When the target is a greater-than-inequality and the prediction is greater than the target,
    # replace the prediction with the target. Analogous for less-than-inequalities.
    preds = np.where(
        np.logical_and(np.greater(preds, targets), gt_targets),
        targets,
        preds,
    )
    preds = np.where(
        np.logical_and(np.less(preds, targets), lt_targets),
        targets,
        preds,
    )
    return mean_absolute_error(targets, preds)


def accuracy(targets: List[int], preds: Union[List[float], List[List[float]]], threshold: float = 0.5) -> float:
    """
    Computes the accuracy of a binary prediction task using a given threshold for generating hard predictions.

    Alternatively, computes accuracy for a multiclass prediction task by picking the largest probability.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0.
    :return: The computed accuracy.
    """
    hard_preds = compute_hard_predictions(preds)

    return accuracy_score(targets, hard_preds)


def recall_metric(targets: List[int], preds: Union[List[float], List[List[float]]], threshold: float = 0.5) -> float:
    """
    Computes the recall of a binary prediction task using a given threshold for generating hard predictions.

    Alternatively, computes recall for a multiclass prediction task by picking the largest probability.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is considered positive.
    :return: The computed recall.
    """
    hard_preds = compute_hard_predictions(preds)

    return recall_score(targets, hard_preds)


def precision_metric(targets: List[int], preds: Union[List[float], List[List[float]]], threshold: float = 0.5) -> float:
    """
    Computes the precision of a binary prediction task using a given threshold for generating hard predictions.

    Alternatively, computes precision for a multiclass prediction task by picking the largest probability.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is considered positive.
    :return: The computed precision.
    """
    hard_preds = compute_hard_predictions(preds)

    return precision_score(targets, hard_preds)


def balanced_accuracy_metric(targets: List[int], preds: Union[List[float], List[List[float]]],
                             threshold: float = 0.5) -> float:
    """
    Computes the balanced accuracy of a binary prediction task using a given threshold for generating hard predictions.

    Alternatively, computes balanced accuracy for a multiclass prediction task by picking the largest probability.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is considered positive.
    :return: The computed balanced accuracy.
    """
    hard_preds = compute_hard_predictions(preds)

    return balanced_accuracy_score(targets, hard_preds)




def f1_metric(targets: List[int], preds: Union[List[float], List[List[float]]], threshold: float = 0.5) -> float:
    """
    Computes the f1 score of a binary prediction task using a given threshold for generating hard predictions.

    Will calculate for a multiclass prediction task by picking the largest probability.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0.
    :return: The computed f1 score.
    """
    hard_preds = compute_hard_predictions(preds)

    if type(preds[0]) == list:  # multiclass
        score = f1_score(targets, hard_preds, average='micro')
    else:  # binary prediction
        score = f1_score(targets, hard_preds)

    return score


def mcc_metric(targets: List[int], preds: Union[List[float], List[List[float]]], threshold: float = 0.5) -> float:
    """
    Computes the Matthews Correlation Coefficient of a binary prediction task using a given threshold for generating hard predictions.

    Will calculate for a multiclass prediction task by picking the largest probability.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0.
    :return: The computed accuracy.
    """
    hard_preds = compute_hard_predictions(preds)

    return matthews_corrcoef(targets, hard_preds)


def sid_metric(model_spectra: List[List[float]], target_spectra: List[List[float]], threshold: float = None,
               batch_size: int = 50) -> float:
    """
    Metric function for use with spectra data type.

    :param model_spectra: The predicted spectra output from a model with shape (num_data, spectrum_length).
    :param target_spectra: The target spectra with shape (num_data, spectrum_length). Values must be normalized so that each spectrum sums to 1.
        Excluded values in target spectra will have a value of None.
    :param threshold: Function requires that values are positive and nonzero. Values below the threshold will be replaced with the threshold value.
    :param batch_size: Batch size for calculating metric.
    :return: The average SID value for the predicted spectra.
    """
    losses = []
    num_iters, iter_step = len(model_spectra), batch_size

    for i in trange(0, num_iters, iter_step):

        # Create batches
        batch_preds = model_spectra[i:i + iter_step]
        batch_preds = np.array(batch_preds)
        batch_targets = target_spectra[i:i + iter_step]
        batch_mask = np.array([[x is not None for x in b] for b in batch_targets])
        batch_targets = np.array([[1 if x is None else x for x in b] for b in batch_targets])

        # Normalize the model spectra before comparison
        if threshold is not None:
            batch_preds[batch_preds < threshold] = threshold
        batch_preds[~batch_mask] = 0
        sum_preds = np.sum(batch_preds, axis=1, keepdims=True)
        batch_preds = batch_preds / sum_preds

        # Calculate loss value
        batch_preds[~batch_mask] = 1  # losses in excluded regions will be zero because log(1/1) = 0.
        loss = batch_preds * np.log(batch_preds / batch_targets) + batch_targets * np.log(batch_targets / batch_preds)
        loss = np.sum(loss, axis=1)

        # Gather batches
        loss = loss.tolist()
        losses.extend(loss)

    loss = np.mean(loss)

    return loss


def wasserstein_metric(model_spectra: List[List[float]], target_spectra: List[List[float]], threshold: float = None,
                       batch_size: int = 50) -> float:
    """
    Metric function for use with spectra data type. This metric assumes that values are evenly spaced.

    :param model_spectra: The predicted spectra output from a model with shape (num_data, spectrum_length).
    :param target_spectra: The target spectra with shape (num_data, spectrum_length). Values must be normalized so that each spectrum sums to 1.
        Excluded values in target spectra will have value None.
    :param threshold: Function requires that values are positive and nonzero. Values below the threshold will be replaced with the threshold value.
    :param batch_size: Batch size for calculating metric.
    :return: The average wasserstein loss value for the predicted spectra.
    """
    losses = []
    num_iters, iter_step = len(model_spectra), batch_size

    for i in trange(0, num_iters, iter_step):

        # Create batches
        batch_preds = model_spectra[i:i + iter_step]
        batch_preds = np.array(batch_preds)
        batch_targets = target_spectra[i:i + iter_step]
        batch_mask = np.array([[x is not None for x in b] for b in batch_targets])
        batch_targets = np.array([[0 if x is None else x for x in b] for b in batch_targets])

        # Normalize the model spectra before comparison
        if threshold is not None:
            batch_preds[batch_preds < threshold] = threshold
        batch_preds[~batch_mask] = 0
        sum_preds = np.sum(batch_preds, axis=1, keepdims=True)
        batch_preds = batch_preds / sum_preds

        # Calculate loss value
        target_cum = np.cumsum(batch_targets, axis=1)
        preds_cum = np.cumsum(batch_preds, axis=1)
        loss = np.abs(target_cum - preds_cum)
        loss = np.sum(loss, axis=1)

        # Gather batches
        loss = loss.tolist()
        losses.extend(loss)

    loss = np.mean(loss)

    return loss

# def squared_log_fugacity_difference_metric(
#         pred_values: List[List[float]],
#         targets: List[List[float]],
#         hybrid_model_features: List[List[float]],
#         vle_inf_dilution: bool,
# ) -> float:
#     pred_y1, pred_y2, pred_log10P, gamma_1, gamma_2, log10p1sat, log10p2sat = np.split(np.array(pred_values), 7, axis=1)

#     y1 = np.array([[x[0]] for x in targets])
#     y2 = np.array([[x[1]] for x in targets])
#     log10P = np.array([[x[2]] for x in targets])
#     x1 = np.array([[x[0]] for x in hybrid_model_features])
#     x2 = np.array([[x[1]] for x in hybrid_model_features])
#     if vle_inf_dilution:
#         gamma_1_inf = np.array([x[3] for x in targets])
#     # make mask
#     x1_not_zero = x1 != 0
#     x2_not_zero = x2 != 0
#     if vle_inf_dilution:
#         g1_inf_not_nan = ~np.isnan(gamma_1_inf)
#         mask = np.concatenate([
#             x1_not_zero,
#             x2_not_zero & ~g1_inf_not_nan,
#             g1_inf_not_nan,
#         ], axis=1, dtype=bool)
#     else:
#         mask = np.concatenate([
#             x1_not_zero,
#             x2_not_zero,
#         ], axis=1, dtype=bool)
#     # apply mask on inputs
#     x1 = np.where(mask[:, [0]], x1, np.ones_like(x1))
#     x2 = np.where(mask[:, [1]], x2, np.ones_like(x2))
#     y1 = np.where(mask[:, [0]], y1, np.ones_like(y1))
#     y2 = np.where(mask[:, [1]], y2, np.ones_like(y2))
#     loss_1 = (np.log10(x1 * gamma_1 / y1) + log10p1sat - log10P) ** 2
#     loss_2 = (np.log10(x2 * gamma_2 / y2) + log10p2sat - log10P) ** 2
#     if vle_inf_dilution:
#         loss_inf = (np.log10(gamma_1_inf / gamma_1)) ** 2
#         loss = np.concatenate((loss_1, loss_2, loss_inf), axis=1)
#     else:
#         loss = np.concatenate((loss_1, loss_2), axis=1)
#     loss[~mask] = np.nan
#     return np.nanmean(loss)