from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "Baseline"

_C.MODEL.FREEZE_LAYERS = []

# MoCo memory size
_C.MODEL.QUEUE_SIZE = 8192

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

_C.MODEL.BACKBONE.NAME = "build_resnet_backbone"
_C.MODEL.BACKBONE.DEPTH = "50x"
_C.MODEL.BACKBONE.LAST_STRIDE = 1
# Backbone feature dimension
_C.MODEL.BACKBONE.FEAT_DIM = 2048
# Normalization method for the convolution layers.
_C.MODEL.BACKBONE.NORM = "BN"
# If use IBN block in backbone
_C.MODEL.BACKBONE.WITH_IBN = False
# If use SE block in backbone
_C.MODEL.BACKBONE.WITH_SE = False
# If use Non-local block in backbone
_C.MODEL.BACKBONE.WITH_NL = False
# Vision Transformer options
_C.MODEL.BACKBONE.SIE_COE = 3.0
_C.MODEL.BACKBONE.STRIDE_SIZE = (16, 16)
_C.MODEL.BACKBONE.DROP_PATH_RATIO = 0.1
_C.MODEL.BACKBONE.DROP_RATIO = 0.0
_C.MODEL.BACKBONE.ATT_DROP_RATE = 0.0
# If use ImageNet pretrain model
_C.MODEL.BACKBONE.PRETRAIN = False
# Pretrain model path
_C.MODEL.BACKBONE.PRETRAIN_PATH = ''

# ---------------------------------------------------------------------------- #
# REID HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.HEADS = CN()
_C.MODEL.HEADS.NAME = "EmbeddingHead"
# Normalization method for the convolution layers.
_C.MODEL.HEADS.NORM = "BN"
# Number of identity
_C.MODEL.HEADS.NUM_CLASSES = 0
# Embedding dimension in head
_C.MODEL.HEADS.EMBEDDING_DIM = 0
# If use BNneck in embedding
_C.MODEL.HEADS.WITH_BNNECK = False
# Triplet feature using feature before(after) bnneck
_C.MODEL.HEADS.NECK_FEAT = "before"  # options: before, after
# Pooling layer type
_C.MODEL.HEADS.POOL_LAYER = "GlobalAvgPool"

# Classification layer type
_C.MODEL.HEADS.CLS_LAYER = "Linear"  # ArcSoftmax" or "CircleSoftmax"

# Margin and Scale for margin-based classification layer
_C.MODEL.HEADS.MARGIN = 0.
_C.MODEL.HEADS.SCALE = 1

# ---------------------------------------------------------------------------- #
# REID LOSSES options
# ---------------------------------------------------------------------------- #
_C.MODEL.LOSSES = CN()
_C.MODEL.LOSSES.NAME = ("CrossEntropyLoss",)

# Cross Entropy Loss options
_C.MODEL.LOSSES.CE = CN()
# if epsilon == 0, it means no label smooth regularization,
# if epsilon == -1, it means adaptive label smooth regularization
_C.MODEL.LOSSES.CE.EPSILON = 0.0
_C.MODEL.LOSSES.CE.ALPHA = 0.2
_C.MODEL.LOSSES.CE.SCALE = 1.0

# Focal Loss options
_C.MODEL.LOSSES.FL = CN()
_C.MODEL.LOSSES.FL.ALPHA = 0.25
_C.MODEL.LOSSES.FL.GAMMA = 2
_C.MODEL.LOSSES.FL.SCALE = 1.0

# Triplet Loss options
_C.MODEL.LOSSES.TRI = CN()
_C.MODEL.LOSSES.TRI.MARGIN = 0.3
_C.MODEL.LOSSES.TRI.NORM_FEAT = False
_C.MODEL.LOSSES.TRI.HARD_MINING = False
_C.MODEL.LOSSES.TRI.SCALE = 1.0

# Circle Loss options
_C.MODEL.LOSSES.CIRCLE = CN()
_C.MODEL.LOSSES.CIRCLE.MARGIN = 0.25
_C.MODEL.LOSSES.CIRCLE.GAMMA = 128
_C.MODEL.LOSSES.CIRCLE.SCALE = 1.0

# Cosface Loss options
_C.MODEL.LOSSES.COSFACE = CN()
_C.MODEL.LOSSES.COSFACE.MARGIN = 0.25
_C.MODEL.LOSSES.COSFACE.GAMMA = 128
_C.MODEL.LOSSES.COSFACE.SCALE = 1.0

# Path to a checkpoint file to be loaded to the model. You can find available models in the model zoo.
_C.MODEL.WEIGHTS = ""

# Values to be used for image normalization
_C.MODEL.PIXEL_MEAN = [0.485*255, 0.456*255, 0.406*255]
# Values to be used for image normalization
_C.MODEL.PIXEL_STD = [0.229*255, 0.224*255, 0.225*255]

# -----------------------------------------------------------------------------
# KNOWLEDGE DISTILLATION
# -----------------------------------------------------------------------------

_C.KD = CN()
_C.KD.MODEL_CONFIG = []
_C.KD.MODEL_WEIGHTS = []
_C.KD.EMA = CN({"ENABLED": False})
_C.KD.EMA.MOMENTUM = 0.999

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]

# `True` if cropping is used for data augmentation during training
_C.INPUT.CROP = CN({"ENABLED": False})
# Size of the image cropped
_C.INPUT.CROP.SIZE = [224, 224]
# Size of the origin size cropped
_C.INPUT.CROP.SCALE = [0.16, 1]
# Aspect ratio of the origin aspect ratio cropped
_C.INPUT.CROP.RATIO = [3./4., 4./3.]

# Random probability for image horizontal flip
_C.INPUT.FLIP = CN({"ENABLED": False})
_C.INPUT.FLIP.PROB = 0.5

# Value of padding size
_C.INPUT.PADDING = CN({"ENABLED": False})
_C.INPUT.PADDING.MODE = 'constant'
_C.INPUT.PADDING.SIZE = 10

# Random color jitter
_C.INPUT.CJ = CN({"ENABLED": False})
_C.INPUT.CJ.PROB = 0.5
_C.INPUT.CJ.BRIGHTNESS = 0.15
_C.INPUT.CJ.CONTRAST = 0.15
_C.INPUT.CJ.SATURATION = 0.1
_C.INPUT.CJ.HUE = 0.1

# Random Affine
_C.INPUT.AFFINE = CN({"ENABLED": False})

# Auto augmentation
_C.INPUT.AUTOAUG = CN({"ENABLED": False})
_C.INPUT.AUTOAUG.PROB = 0.0

# Augmix augmentation
_C.INPUT.AUGMIX = CN({"ENABLED": False})
_C.INPUT.AUGMIX.PROB = 0.0

# Random Erasing
_C.INPUT.REA = CN({"ENABLED": False})
_C.INPUT.REA.PROB = 0.5
_C.INPUT.REA.VALUE = [0.485*255, 0.456*255, 0.406*255]
# Random Patch
_C.INPUT.RPT = CN({"ENABLED": False})
_C.INPUT.RPT.PROB = 0.5

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training
_C.DATASETS.NAMES = ("Market1501",)
# List of the dataset names for testing
_C.DATASETS.TESTS = ("Market1501",)
# Combine trainset and testset joint training
_C.DATASETS.COMBINEALL = False

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Options: TrainingSampler, NaiveIdentitySampler, BalancedIdentitySampler
_C.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
# Number of instance for each person
_C.DATALOADER.NUM_INSTANCE = 4
_C.DATALOADER.NUM_WORKERS = 8

# For set re-weight
_C.DATALOADER.SET_WEIGHT = []

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

# AUTOMATIC MIXED PRECISION
_C.SOLVER.AMP = CN({"ENABLED": False})

# Optimizer
_C.SOLVER.OPT = "Adam"

_C.SOLVER.MAX_EPOCH = 120

_C.SOLVER.BASE_LR = 3e-4

# This LR is applied to the last classification layer if
# you want to 10x higher than BASE_LR.
_C.SOLVER.HEADS_LR_FACTOR = 1.

_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.NESTEROV = False

_C.SOLVER.WEIGHT_DECAY = 0.0005
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0005

# The previous detection code used a 2x higher LR and 0 WD for bias.
# This is not useful (at least for recent models). You should avoid
# changing these and they exists only to reproduce previous model
# training if desired.
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY

# Multi-step learning rate options
_C.SOLVER.SCHED = "MultiStepLR"

_C.SOLVER.DELAY_EPOCHS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = [30, 55]

# Cosine annealing learning rate options
_C.SOLVER.ETA_MIN_LR = 1e-7

# Warmup options
_C.SOLVER.WARMUP_FACTOR = 0.1
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"

# Backbone freeze iters
_C.SOLVER.FREEZE_ITERS = 0

_C.SOLVER.CHECKPOINT_PERIOD = 20

# Number of images per batch across all machines.
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 256, each GPU will
# see 32 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# Gradient clipping
_C.SOLVER.CLIP_GRADIENTS = CN({"ENABLED": False})
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
# Maximum absolute value used for clipping gradients
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 5.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

_C.TEST = CN()

_C.TEST.EVAL_PERIOD = 20

# Number of images per batch across all machines.
_C.TEST.IMS_PER_BATCH = 64
_C.TEST.METRIC = "cosine"
_C.TEST.ROC = CN({"ENABLED": False})
_C.TEST.FLIP = CN({"ENABLED": False})

# Average query expansion
_C.TEST.AQE = CN({"ENABLED": False})
_C.TEST.AQE.ALPHA = 3.0
_C.TEST.AQE.QE_TIME = 1
_C.TEST.AQE.QE_K = 5

# Re-rank
_C.TEST.RERANK = CN({"ENABLED": False})
_C.TEST.RERANK.K1 = 20
_C.TEST.RERANK.K2 = 6
_C.TEST.RERANK.LAMBDA = 0.3

# Precise batchnorm
_C.TEST.PRECISE_BN = CN({"ENABLED": False})
_C.TEST.PRECISE_BN.DATASET = 'Market1501'
_C.TEST.PRECISE_BN.NUM_ITER = 300

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "logs/"

# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = False
