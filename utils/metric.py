import os
import numpy as np
from matplotlib import pyplot as plt
from IPython import embed


def calc_accuracy(pred_label, gt_label):
    """
    :param pred_label: A list contains of predicted labels in type np.ndarray.
    :param gt_label:  A list contains ground truths.
    :return: A number between 0 and 1 representing the accuracy rate.
    """
    pred = np.concatenate(pred_label).reshape(-1)
    gt = np.concatenate(gt_label).reshape(-1)
    return float((pred == gt).sum() / np.ones_like(gt).sum())


def draw_loss_curve(args, loss_list):
    """
    :param args: Runtime arguments.
    :param loss_list: A list contains of losses of float number across different batches.
    :return: None, draw and save a curve under `{args.save_path} / {args.task_name} / loss.png`
    """
    os.makedirs(f"{args.save_path}/{args.task_name}", exist_ok=True)
    plt.cla()
    # TODO Start: Plot curve using values in loss_list #
    plt.plot(range(0, len(loss_list)), loss_list)

    # TODO End #
    plt.savefig(f"{args.save_path}/{args.task_name}/loss.png")
