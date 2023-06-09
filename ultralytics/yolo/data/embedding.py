import lancedb
import pyarrow as pa
import numpy as np
import cv2
import torch.nn.functional as F
from sklearn.decomposition import PCA
from pathlib import Path
from lancedb.embeddings import with_embeddings

from ultralytics import YOLO
from ultralytics.yolo.data.utils import check_cls_dataset, check_det_dataset, IMG_FORMATS
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
            embeddings = embeddings if isinstance(embeddings, list) else [embeddings]
            yield from embeddings


def get_train_split(data='coco128.yaml', task='detect', batch=16):
    # TODO: handle exception
    if task == 'classify':
        data = check_cls_dataset(data)
    elif data.endswith('.yaml') or task in ('detect', 'segment'):
        data = check_det_dataset(data)

    return data['train']


class DatasetUtil:
    """
    Dataset utils. Supports detection, segmnetation and Pose and YOLO models.
    """
    def __init__(self, data, project=None, verbose=False) -> None:
        """
        Args:
            dataset (str): path to dataset
            model (str, optional): path to model. Defaults to None.
        """
        self.data = data
        self.project = project or "runs/dataset"
        self.verbose = verbose
        self.predictor = None
        self.trainset = None
        self.orig_imgs = None
        self.table = None
            
    def build_embeddings(self, model=None):
        self.model = YOLO(model)
        trainset = get_train_split(self.data, task=self.model.task)
        trainset = trainset if isinstance(trainset, list) else [trainset]
        self.trainset = trainset
        self.predictor = EmbeddingsPredictor()
        if self.table is not None:
            LOGGER.info("Overwriting the existing embedding space")
        
        self.orig_imgs = []
        for train_split in trainset:
            print(train_split)
            path = Path(train_split)
            files = sorted(str(x) for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
            self.orig_imgs.extend(files)
        
        db = self._connect()
        table = pa.table([self.orig_imgs], names=["path"]).to_pandas()
        self.table = with_embeddings(self._embedding_func, table, "path")
    
    def _connect(self):
        db = lancedb.connect(self.project)

        return db
        
    def _embedding_func(self, imgs):
        embed_gen = self.predictor(imgs, model=self.model.model, stream=True) # generator
        for i, embed in enumerate(embed_gen):
            import pdb;pdb.set_trace()

        return [emb.squeeze().cpu().numpy() for emb in embed_gen]


        
        

#build_table("VOC.yaml")
#db = lancedb.connect("db/")
#table = db.open_table("VOC")
#project_embeddings(table)

ds = DatasetUtil("coco128.yaml")
ds.build_embeddings("yolov8n.pt")


'''

def create_mosaic(image_paths):
    image_sizes = [cv2.imread(path).shape for path in image_paths]

    mosaic_width = np.sum(image_sizes[:, 0])
    mosaic_height = np.sum(image_sizes[:, 1])

    mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.uint8)

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (image_sizes[i][1], image_sizes[i][0]))
        mosaic[i * image_sizes[i][1]:(i + 1) * image_sizes[i][1], :image_sizes[i][0]] = image

    # Display the mosaic
    cv2.imshow('Mosaic', mosaic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_similar_imgs(img_path, table, n=10):
    predictor = EmbeddingsPredictor()
    embeddings = list(predictor(img_path, model='yolov8n.pt', stream=True))[0]
    embeddings = embeddings[0].squeeze().cpu().numpy()
    sim = table.search(embeddings).limit(n).to_df()["path"].to_pylist()
    return sim

PREDICTOR = EmbeddingsPredictor
MODEL = None
#def emnedding_func(imgs):

def build_table(data='VOC.yaml', model='yolov8n.pt'):
    global PREDICTOR
    global MODEL
    model = YOLO(model)
    predictor = EmbeddingsPredictor()
    path_col = [] # path
    trainset = get_train_split(data, task=model.task)
    trainset = trainset if isinstance(trainset, list) else [trainset]
    for train_split in trainset:
        path = Path(train_split)
        files = sorted(x for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
        path_col.extend(files)

        embeddings = predictor(train_split, model=model.model, stream=True)
        for embedding in embeddings:
            print(embedding[0])
            cols[0].append(embedding[0].squeeze().cpu().numpy())
            cols[1].append(embedding[1][0])
    db = lancedb.connect(f"db/")
    table = pa.table(path_col, names=["path"])
    table = db.create_table(name=data.split('.')[0], data=table, mode="overwrite")

    return table


def project_embeddings(table, n_components=2):
    pca = PCA(n_components=n_components)
    embeddings = np.array(table.to_arrow()['vector'].to_pylist())
    embeddings_reduced = pca.fit_transform(embeddings)
    # TODO: plot embeddings_reduced
    import wandb
    wandb.init(project="lance")
    table = wandb.Table(data=embeddings_reduced, columns=["x", "y"])
    wandb.log({
    "embeddings": wandb.Table(
        columns = ["D1", "D2"], 
        data    = embeddings_reduced
    )
    })
    wandb.finish()


'''