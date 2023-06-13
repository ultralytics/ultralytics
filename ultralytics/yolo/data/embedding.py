from pathlib import Path

import cv2
import lancedb
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import torch.nn.functional as F
import yaml
from lancedb.embeddings import with_embeddings
from sklearn.decomposition import PCA
from tqdm import tqdm

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


def get_dataset_info(data='coco128.yaml', task='detect', batch=16):
    # TODO: handle exception
    if task == 'classify':
        data = check_cls_dataset(data)
    elif data.endswith('.yaml') or task in ('detect', 'segment'):
        data = check_det_dataset(data)

    return data


class DatasetUtil:
    """
    Dataset utils. Supports detection, segmnetation and Pose and YOLO models.
    """

    #TODO: Allow starting from an existing table
    def __init__(self, data=None, table=None, project=None) -> None:
        """
        Args:
            dataset (str): path to dataset
            model (str, optional): path to model. Defaults to None.
        """
        if data is None and table is None:
            raise ValueError('Either data or table must be provided')

        self.data = data 
        self.table = None
        self.project = project or 'runs/dataset'
        self.table_name = data if data is not None else Path(table).stem # Keep the table name when copying
        self.temp_table_name = self.table_name + '_temp'
        self.dataset_info = None
        self.predictor = None
        self.trainset = None
        self.orig_imgs = []
        self.removed_imgs = []
        self.verbose = False  # For embedding function

        # copy table to project if table is provided
        if table:
            self.table = self._copy_table_to_project(table)

    def build_embeddings(self, model="yolov8n.pt", verbose=False, force=False):
        self.model = YOLO(model)
        self.dataset_info = get_dataset_info(self.data, task=self.model.task)
        trainset = self.dataset_info['train']
        trainset = trainset if isinstance(trainset, list) else [trainset]
        self.trainset = trainset
        self.predictor = EmbeddingsPredictor()
        self.predictor.setup_model(self.model.model)
        self.verbose = verbose

        self.orig_imgs = []
        for train_split in trainset:
            print(train_split)
            path = Path(train_split)
            files = sorted(str(x) for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
            self.orig_imgs.extend(files)

        db = self._connect()
        if not force and self.data in db.table_names():
            LOGGER.info('Embedding space already exists. Attempting to reuse it. Use force=True to overwrite.')
            self.table = db.open_table(self.data)
            if len(self.table.to_arrow()) == len(self.orig_imgs):
                return
            else:
                LOGGER.info('Table length does not match the number of images in the dataset. Building embeddings...')

        idx = [i for i in range(len(self.orig_imgs))]  # for easier hashing #TODO: remove. not needed anymore
        df = pa.table([self.orig_imgs, idx], names=['path', 'id']).to_pandas()
        pa_table = with_embeddings(self._embedding_func, df, 'path')
        self.table = db.create_table(self.table_name, data=pa_table, mode='overwrite')

    def project_embeddings(self, n_components=2):
        if self.table is None:
            LOGGER.error('No embedding space found. Please build the embedding space first.')
            return None
        pca = PCA(n_components=n_components)
        embeddings = np.array(self.table.to_arrow['vector'].to_pylist())
        embeddings_reduced = pca.fit_transform(embeddings)

        return embeddings_reduced

    def get_similar_imgs(self, img, n=10):
        if isinstance(img, int):
            img_path = self.orig_imgs[img]
        elif isinstance(img, (str, Path)):
            img_path = img
        else:
            LOGGER.error('img should be index from the table(int) or path of an image (str or Path)')
            return
        # predictor = EmbeddingsPredictor()
        embeddings = self.predictor.embed(img_path).squeeze().cpu().numpy()
        sim = self.table.search(embeddings).limit(n).to_df()
        return sim['path'], sim['id']

    def plot_similar_imgs(self, img=None, n=10):
        img_paths, _ = self.get_similar_imgs(img, n)
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
        # Display the grid of images
        plt.show()

    def get_similarity_index(self, sim_thres=0.9, top_k=0.01):
        """

        Args:
            sim_thres (float, optional): Similarity threshold to set the minimum similarity. Defaults to 0.9.
            top_k (float, optional): Top k fraction of the similar embeddings to apply the threshold on. Defaults to 0.1.
        """
        if self.table is None:
            LOGGER.error('No embedding space found. Please build the embedding space first.')
            return None
        if top_k > 1.0:
            LOGGER.warning('top_k should be between 0 and 1. Setting top_k to 1.0')
            top_k = 1.0
        if top_k < 0.0:
            LOGGER.warning('top_k should be between 0 and 1. Setting top_k to 0.0')
            top_k = 0.0
        if sim_thres > 1.0:
            LOGGER.warning('sim_thres should be between 0 and 1. Setting sim_thres to 1.0')
            sim_thres = 1.0
        if sim_thres < 0.0:
            LOGGER.warning('sim_thres should be between 0 and 1. Setting sim_thres to 0.0')
            sim_thres = 0.0

        threshold = 1.0 - sim_thres
        embs = np.array(self.table.to_arrow()['vector'].to_pylist())
        index = np.zeros(len(embs))
        limit = int(len(embs) * top_k)
        for _, emb in enumerate(tqdm(embs)):
            df = self.table.search(emb).metric('cosine').limit(limit).to_df().query(f'score <= {threshold}')
            for idx in df['id'][1:]:
                index[idx] += 1
        self.sim_index = index
        return index

    def plot_similirity_index(self, threshold=0.9, sorted=True):
        index = self.get_similarity_index(threshold)
        if sorted:
            index = np.sort(index)
        plt.bar([i for i in range(len(index))], index)
        plt.xlabel('idx')
        plt.ylabel('similarity count')
        plt.show()

    def remove_imgs(self, idxs):
        """
        Works on temporary table. To apply the changes to the main table, call `persist()`

        Args:
            idxs (int or list): Index of the image to remove from the dataset.
        """
        if isinstance(idxs, int):
            idxs = [idxs]

        pa_table = self.table.to_arrow()
        mask = [True for _ in range(len(pa_table))]
        for idx in idxs:
            mask[idx] = False
            self.removed_imgs.append(self.orig_imgs.pop(idx))
        ids = [i for i in range(len(self.orig_imgs))]
        table = pa_table.filter(mask).set_column(1, 'id', [ids])

        db = self._connect()
        self.table = db.create_table(self.temp_table_name, data=table, mode='overwrite')  # work on a temporary table
        self.log_status()

    def reset(self):
        """
        Resets the dataset to the original state.
        """
        if self.table is None:
            LOGGER.info('No changes made to the dataset.')
            return

        db = self._connect()
        if self.temp_table_name in db.table_names():
            db.drop_table(self.temp_table_name)

        self.table = db.open_table(self.data)
        self.orig_imgs = self.table.to_arrow()['path'].to_pylist()
        self.removed_imgs = []
        LOGGER.info('Dataset reset to original state.')

    def persist(self, name=None):
        """
        Persists the changes made to the dataset.
        """
        db = self._connect()
        if self.table is None or self.temp_table_name not in db.table_names():
            LOGGER.info('No changes made to the dataset.')
            return

        LOGGER.info('Persisting changes to the dataset...')
        self.log_status()

        if not name:
            name = self.data.split('.')[0] + '_updated.' + self.data.split('.')[1]
        train_txt = 'train_updated.txt'

        path = Path(self.dataset_info['path']).resolve()  # add new train.txt file in the dataset parent path
        if (path / train_txt).exists():
            (path / train_txt).unlink()  # remove existing

        for img in tqdm(self.orig_imgs):
            with open(path / train_txt, 'a') as f:
                f.write(f'./{Path(img).relative_to(path).as_posix()}' + '\n')  # add image to txt file

        new_dataset_info = self.dataset_info.copy()
        new_dataset_info['train'] = train_txt
        for key, value in new_dataset_info.items():
            if isinstance(value, Path):
                new_dataset_info[key] = value.as_posix()

        yaml.dump(new_dataset_info, open(Path(self.project) / name, 'w'))  # update dataset.yaml file

        # TODO: not sure if this should be called data_final to prevent overwriting the original data? Creating embs for large datasets is expensive
        self.table = db.create_table(self.table_name, data=self.table.to_arrow(), mode='overwrite')
        db.drop_table(self.temp_table_name)

        LOGGER.info('Changes persisted to the dataset.')
        self._log_training_cmd((Path(self.project) / name).as_posix())

    def log_status(self):
        # TODO: Pretty print log status
        LOGGER.info(f'Number of images: {len(self.orig_imgs)}')
        LOGGER.info(f'Number of removed images: {len(self.removed_imgs)}')
        LOGGER.info(f'Number of images in the embedding space: {len(self.table)}')

    def _log_training_cmd(self, data_path):
        LOGGER.info('New dataset created successfully! Run the following command to train a model:')
        LOGGER.info(f'yolo train data={data_path} epochs=10')

    def _connect(self):
        db = lancedb.connect(self.project)

        return db
    
    def _create_table(self, name, data=None, mode='overwrite'):
        db = lancedb.connect(self.project)
        table = db.create_table(name, data=data, mode=mode)

        return table
    
    def _open_table(self, name):
        db = lancedb.connect(self.project)
        table = db.open_table(name)

        return table
    
    def _copy_table_to_project(self, table_path):
        if not table_path.endswith(".lance"):
            raise ValueError("Table must be a .lance file")

        LOGGER.info(f"Copying table from {table_path}")
        path = Path(table_path).parent
        name = Path(table_path).stem # lancedb doesn't need .lance extension
        db = lancedb.connect(path)
        table = db.open_table(name)
        return self._create_table(self.table_name, data=table.to_arrow(), mode='overwrite')

    def _embedding_func(self, imgs):
        embeddings = []
        for img in tqdm(imgs):
            if self.verbose:
                LOGGER.info(img)
            embeddings.append(self.predictor.embed(img, verbose=self.verbose).squeeze().cpu().numpy())
        return embeddings

    def _create_trainable_dataset(self, train_txt):
        pass

    def create_index(self):
        # TODO: create index
        pass


#build_table("VOC.yaml")
#db = lancedb.connect("db/")
#table = db.open_table("VOC")
#project_embeddings(table)
project = "runs/test/temp/"
ds = DatasetUtil("coco8.yaml", project=project)
ds.build_embeddings()
table = project + ds.table_name + ".lance"
ds2 = DatasetUtil(table=table)
"""
ds = DatasetUtil('coco8.yaml')
ds.build_embeddings('yolov8n.pt')

#ds.plot_similar_imgs(4, 10)
#ds.plot_similirity_index()
sim = ds.get_similarity_index()
paths, ids = ds.get_similar_imgs(4, 10)
ds.remove_imgs(ids)
ds.reset()
ds.log_status()
ds.remove_imgs([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
ds.remove_imgs([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

ds.persist()
"""