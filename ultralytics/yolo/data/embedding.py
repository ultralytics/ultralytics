import lancedb
import pyarrow as pa
import numpy as np
import cv2
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
from lancedb.embeddings import with_embeddings
from sklearn.decomposition import PCA

from ultralytics import YOLO
from ultralytics.yolo.data.utils import IMG_FORMATS, check_cls_dataset, check_det_dataset
from ultralytics.yolo.utils import LOGGER, ops
from ultralytics.yolo.utils.checks import check_requirements
from ultralytics.yolo.utils.torch_utils import smart_inference_mode
from ultralytics.yolo.v8.detect.predict import DetectionPredictor


class EmbeddingsPredictor(DetectionPredictor):

    def postprocess(self, preds, img, orig_imgs):
        embedding = preds[1]
        embedding = F.adaptive_avg_pool2d(embedding, 2).flatten(1)
        return embedding

    @smart_inference_mode()
    def embed(self, source=None, model=None, verbose=True):
        """Streams real-time inference on camera feed and saves results to file."""
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
            if verbose:
                LOGGER.info(path[0])
            # Preprocess
            with profilers[0]:
                im = self.preprocess(im0s)

            # Inference
            with profilers[1]:
                preds = self.model(im, augment=self.args.augment, embed_from=-1)

            with profilers[2]:
                embeddings = self.postprocess(preds, im, im0s)

            return embeddings
            # yeilding seems pointless as this is designed specifically for large datasets,
            # batching with embed_func would make things complex


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
    #TODO: Allow starting from an existing table
    def __init__(self, data, table=None, project=None, verbose=False) -> None:
        """
        Args:
            dataset (str): path to dataset
            model (str, optional): path to model. Defaults to None.
        """
        self.data = data
        self.project = project or 'runs/dataset'
        self.verbose = verbose
        self.predictor = None
        self.trainset = None
        self.orig_imgs = None
        self.table = table
            
    def build_embeddings(self, model=None):
        self.model = YOLO(model)
        trainset = get_train_split(self.data, task=self.model.task)
        trainset = trainset if isinstance(trainset, list) else [trainset]
        self.trainset = trainset
        self.predictor = EmbeddingsPredictor()
        self.predictor.setup_model(self.model.model)
        if self.table is not None:
            LOGGER.info('Overwriting the existing embedding space')

        self.orig_imgs = []
        for train_split in trainset:
            print(train_split)
            path = Path(train_split)
            files = sorted(str(x) for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
            self.orig_imgs.extend(files)

        db = self._connect()
        pa_table = pa.table([self.orig_imgs], names=["path"]).to_pandas()
        pa_table = with_embeddings(self._embedding_func, pa_table, "path")
        self.table = db.create_table(self.data, data=pa_table, mode="overwrite") # TODO: reuse built table

    def project_embeddings(self, n_components=2):
        if self.table is None:
            LOGGER.error("No embedding space found. Please build the embedding space first.")
            return None
        pca = PCA(n_components=n_components)
        embeddings = np.array(self.table.to_arrow()["vector"].to_pylist())
        embeddings_reduced = pca.fit_transform(embeddings)

        return embeddings_reduced
    
    def get_similar_imgs(self, img, n=10):
        if isinstance(img, int):
            img_path = self.orig_imgs[img]
        elif isinstance(img, (str, Path)):
            img_path = img
        else:
            LOGGER.error("img should be index from the table(int) or path of an image (str or Path)")
            return
        # predictor = EmbeddingsPredictor()
        embeddings = self.predictor.embed(img_path).squeeze().cpu().numpy()
        sim = self.table.search(embeddings).limit(n).to_df()["path"]
        return sim

    def show_similar_imgs(self, img=None, n=10):        
        img_paths = self.get_similar_imgs(img, n)
        images = [cv2.imread(image_path) for image_path in img_paths]

        # Resize the images to the minimum and maximum width and height
        resized_images = []
        for image in images:
            resized_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized_images.append(resized_image)

        # Create a grid of the images
        fig, axes = plt.subplots(nrows=len(resized_images) // 2, ncols=2)
        for i, ax in enumerate(axes.ravel()):
            ax.imshow(resized_images[i])
            ax.axis('off')
        import pdb;pdb.set_trace()
        # Display the grid of images
        plt.show()
  

    def _connect(self):
        db = lancedb.connect(self.project)

        return db

    def _embedding_func(self, imgs):

        return [self.predictor.embed(img, verbose=self.verbose).squeeze().cpu().numpy() for img in imgs]


#build_table("VOC.yaml")
#db = lancedb.connect("db/")
#table = db.open_table("VOC")
#project_embeddings(table)

ds = DatasetUtil("coco128.yaml")
ds.build_embeddings("yolov8n.pt")
ds.show_similar_imgs(100, 20)
