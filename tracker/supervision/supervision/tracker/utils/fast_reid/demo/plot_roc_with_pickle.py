# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import matplotlib.pyplot as plt
import sys

sys.path.append('.')
from supervision.tracker.utils.fast_reid.fastreid.utils.visualizer import Visualizer

if __name__ == "__main__":
    baseline_res = Visualizer.load_roc_info("logs/duke_vis/roc_info.pickle")
    mgn_res = Visualizer.load_roc_info("logs/mgn_duke_vis/roc_info.pickle")

    fig = Visualizer.plot_roc_curve(baseline_res['fpr'], baseline_res['tpr'], name='baseline')
    Visualizer.plot_roc_curve(mgn_res['fpr'], mgn_res['tpr'], name='mgn', fig=fig)
    plt.savefig('roc.jpg')

    fig = Visualizer.plot_distribution(baseline_res['pos'], baseline_res['neg'], name='baseline')
    Visualizer.plot_distribution(mgn_res['pos'], mgn_res['neg'], name='mgn', fig=fig)
    plt.savefig('dist.jpg')
