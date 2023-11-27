import sys
import timeit
import numpy as np
import os.path as osp

sys.path.insert(0, osp.dirname(osp.abspath(__file__)) + '/../../..')

from supervision.tracker.utils.fast_reid.fastreid.evaluation.rank import evaluate_rank
from supervision.tracker.utils.fast_reid.fastreid.evaluation.roc import evaluate_roc

"""
Test the speed of cython-based evaluation code. The speed improvements
can be much bigger when using the real reid data, which contains a larger
amount of query and gallery images.
Note: you might encounter the following error:
  'AssertionError: Error: all query identities do not appear in gallery'.
This is normal because the inputs are random numbers. Just try again.
"""

print('*** Compare running time ***')

setup = '''
import sys
import os.path as osp
import numpy as np
sys.path.insert(0, osp.dirname(osp.abspath(__file__)) + '/../../..')
from supervision.tracker.utils.fast_reid.fastreid.evaluation.rank import evaluate_rank
from supervision.tracker.utils.fast_reid.fastreid.evaluation.roc import evaluate_roc
num_q = 30
num_g = 300
dim = 512
max_rank = 5
q_feats = np.random.rand(num_q, dim).astype(np.float32) * 20
q_feats = q_feats / np.linalg.norm(q_feats, ord=2, axis=1, keepdims=True)
g_feats = np.random.rand(num_g, dim).astype(np.float32) * 20
g_feats = g_feats / np.linalg.norm(g_feats, ord=2, axis=1, keepdims=True)
distmat = 1 - np.dot(q_feats, g_feats.transpose())
q_pids = np.random.randint(0, num_q, size=num_q)
g_pids = np.random.randint(0, num_g, size=num_g)
q_camids = np.random.randint(0, 5, size=num_q)
g_camids = np.random.randint(0, 5, size=num_g)
'''

print('=> Using CMC metric')
pytime = timeit.timeit(
    'evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_cython=False)',
    setup=setup,
    number=20
)
cytime = timeit.timeit(
    'evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_cython=True)',
    setup=setup,
    number=20
)
print('Python time: {} s'.format(pytime))
print('Cython time: {} s'.format(cytime))
print('CMC Cython is {} times faster than python\n'.format(pytime / cytime))

print('=> Using ROC metric')
pytime = timeit.timeit(
    'evaluate_roc(distmat, q_pids, g_pids, q_camids, g_camids, use_cython=False)',
    setup=setup,
    number=20
)
cytime = timeit.timeit(
    'evaluate_roc(distmat, q_pids, g_pids, q_camids, g_camids, use_cython=True)',
    setup=setup,
    number=20
)
print('Python time: {} s'.format(pytime))
print('Cython time: {} s'.format(cytime))
print('ROC Cython is {} times faster than python\n'.format(pytime / cytime))

print("=> Check precision")
num_q = 30
num_g = 300
dim = 512
max_rank = 5
q_feats = np.random.rand(num_q, dim).astype(np.float32) * 20
q_feats = q_feats / np.linalg.norm(q_feats, ord=2, axis=1, keepdims=True)
g_feats = np.random.rand(num_g, dim).astype(np.float32) * 20
g_feats = g_feats / np.linalg.norm(g_feats, ord=2, axis=1, keepdims=True)
distmat = 1 - np.dot(q_feats, g_feats.transpose())
q_pids = np.random.randint(0, num_q, size=num_q)
g_pids = np.random.randint(0, num_g, size=num_g)
q_camids = np.random.randint(0, 5, size=num_q)
g_camids = np.random.randint(0, 5, size=num_g)

cmc_py, mAP_py, mINP_py = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_cython=False)

cmc_cy, mAP_cy, mINP_cy = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_cython=True)

np.testing.assert_allclose(cmc_py, cmc_cy, rtol=1e-3, atol=1e-6)
np.testing.assert_allclose(mAP_py, mAP_cy, rtol=1e-3, atol=1e-6)
np.testing.assert_allclose(mINP_py, mINP_cy, rtol=1e-3, atol=1e-6)
print('Rank results between python and cython are the same!')

scores_cy, labels_cy = evaluate_roc(distmat, q_pids, g_pids, q_camids, g_camids, use_cython=True)
scores_py, labels_py = evaluate_roc(distmat, q_pids, g_pids, q_camids, g_camids, use_cython=False)

np.testing.assert_allclose(scores_cy, scores_py, rtol=1e-3, atol=1e-6)
np.testing.assert_allclose(labels_cy, labels_py, rtol=1e-3, atol=1e-6)
print('ROC results between python and cython are the same!\n')

print("=> Check exact values")
print("mAP = {} \ncmc = {}\nmINP = {}\nScores = {}".format(np.array(mAP_cy), cmc_cy, np.array(mINP_cy), scores_cy))
