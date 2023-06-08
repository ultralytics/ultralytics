import lancedb
import numpy as np
import pyarrow as pa
import torch.nn.functional as F
from sklearn.decomposition import PCA

from ultralytics import YOLO
from ultralytics.yolo.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.yolo.utils import LOGGER, ops
from ultralytics.yolo.utils.checks import check_requirements
from ultralytics.yolo.utils.torch_utils import smart_inference_mode
from ultralytics.yolo.v8.detect.predict import DetectionPredictor


class EmbeddingsPredictor(DetectionPredictor):

    def postprocess(self, preds, img, orig_imgs):
        embeddings = preds[1]
        embeddings = F.adaptive_avg_pool2d(embeddings, 2).flatten(1)

        return embeddings

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None):
        """Streams real-time inference on camera feed and saves results to file."""
        if self.args.verbose:
            LOGGER.info('')

        # Setup model
        if not self.model:
            self.setup_model(model)
        # Setup source every time predict is called
        self.setup_source(source if source is not None else self.args.source)

        # Warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True

        self.seen, self.windows, self.batch, profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile())
        for batch in self.dataset:
            path, im0s, vid_cap, s = batch

            # Preprocess
            with profilers[0]:
                im = self.preprocess(im0s)

            # Inference
            with profilers[1]:
                preds = self.model(im, augment=self.args.augment, embed_from=-1)

            with profilers[2]:
                embeddings = self.postprocess(preds, im, im0s)

            yield embeddings, path


def get_train_split(data='coco128.yaml', task='detect', batch=16):
    # TODO: handle exception
    if task == 'classify':
        data = check_cls_dataset(data)
    elif data.endswith('.yaml') or task in ('detect', 'segment'):
        data = check_det_dataset(data)

    return data['train']


def build_table(data='coco128.yaml', model='yolov8n.pt'):
    model = YOLO(model)
    predictor = EmbeddingsPredictor()
    predictor.imgsz = [640, 640]  # hardcode imgsz
    cols = [[], []]  # path, embeddings

    embeddings = predictor(get_train_split(data, task=model.task), model=model.model, stream=True)
    for embedding in embeddings:
        print(embedding[0])

        cols[0].append(embedding[0].squeeze().cpu())
        cols[1].append(embedding[1])

    db = lancedb.connect(f'db/')
    table = pa.table(cols, names=['vector', 'path'])
    table = db.create_table(name=data.split('.')[0], data=table, mode='overwrite')

    return table


def project_embeddings(table, n_components=2):
    pca = PCA(n_components=n_components)
    embeddings = np.array(table.to_arrow()['vector'].to_pylist())
    embeddings_reduced = pca.fit_transform(embeddings)
    import pdb
    pdb.set_trace()
    # TODO: plot embeddings_reduced


#build_table()
db = lancedb.connect('db/')
table = db.open_table('coco128')
project_embeddings(table)
