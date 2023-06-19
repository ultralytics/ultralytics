import glob
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.yolo.data.utils import IMG_FORMATS, check_det_dataset
from ultralytics.yolo.utils import LOGGER, colorstr, ops
from ultralytics.yolo.utils.torch_utils import smart_inference_mode
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

try:
    import lancedb
    import pyarrow as pa  # dependency of lancedb
    from lancedb.embeddings import with_embeddings
    from sklearn.decomposition import PCA
except ImportError:
    LOGGER.error('Please install lancedb and sklearn to use Explorer - `pip install lancedb sklearn`')


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
            path, im0s, _, _ = batch
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
            # yielding seems pointless as this is designed specifically to be used in for loops,
            # batching with embed_func would make things complex


def get_dataset_info(data='coco128.yaml', task='detect'):
    # TODO: handle other tasks
    data = check_det_dataset(data)

    return data


class Explorer:
    """
    YOLO Explorer. Supports detection, segmnetation and Pose and YOLO models.
    """

    def __init__(self, data=None, table=None, model='yolov8n.pt', device='', project='runs/dataset') -> None:
        """
        Args:
            data (str, optional): path to dataset file
            table (str, optional): path to LanceDB table to load embeddings Table from.
            model (str, optional): path to model. Defaults to None.
            device (str, optional): device to use. Defaults to ''. If empty, uses the default device.
            project (str, optional): path to project. Defaults to "runs/dataset".
        """
        if data is None and table is None:
            raise ValueError('Either data or table must be provided')

        self.data = data
        self.table = None
        self.model = model
        self.project = project
        self.table_name = data if data is not None else Path(table).stem  # Keep the table name when copying
        self.temp_table_name = self.table_name + '_temp'
        self.dataset_info = None
        self.predictor = None
        self.trainset = None
        self.removed_img_count = 0
        self.verbose = False  # For embedding function
        self._sim_index = None

        # copy table to project if table is provided
        if table:
            self.table = self._copy_table_to_project(table)
        if model:
            self.predictor = self._setup_predictor(model, device)
        if data:
            self.dataset_info = get_dataset_info(self.data)

    def build_embeddings(self, verbose=False, force=False):
        """
        Builds the embedding space for the dataset in LanceDB table format

        Args:
            verbose (bool, optional): verbose. Defaults to False.
            force (bool, optional): force rebuild. Defaults to False.
        """
        trainset = self.dataset_info['train']
        trainset = trainset if isinstance(trainset, list) else [trainset]
        self.trainset = trainset
        self.verbose = verbose

        orig_imgs = []
        try:
            f = []  # image files
            for p in trainset:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f'{p} does not exist')
            orig_imgs = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert orig_imgs, f'No images found: {p}'
        except Exception as e:
            raise FileNotFoundError(f'Error loading data from {p}') from e

        db = self._connect()
        if not force and self.table_name in db.table_names():
            LOGGER.info('LanceDB embedding space already exists. Attempting to reuse it. Use force=True to overwrite.')
            self.table = self._open_table(self.table_name)
            if len(self.table.to_arrow()) == len(orig_imgs):
                return
            else:
                LOGGER.info('Table length does not match the number of images in the dataset. Building embeddings...')

        idx = [i for i in range(len(orig_imgs))]  # for easier hashing #TODO: remove. not needed anymore
        df = pa.table([orig_imgs, idx], names=['path', 'id']).to_pandas()
        pa_table = with_embeddings(self._embedding_func, df, 'path', batch_size=10000)  # TODO: remove hardcoding?
        self.table = self._create_table(self.table_name, data=pa_table, mode='overwrite')
        LOGGER.info(f'{colorstr("LanceDB:")} Embedding space built successfully.')

    def plot_embeddings(self):
        """
        Projects the embedding space to 2D using PCA

        Args:
            n_components (int, optional): number of components. Defaults to 2.
        """
        if self.table is None:
            LOGGER.error('No embedding space found. Please build the embedding space first.')
            return None
        pca = PCA(n_components=2)
        embeddings = np.array(self.table.to_arrow()['vector'].to_pylist())
        embeddings = pca.fit_transform(embeddings)
        plt.scatter(embeddings[:, 0], embeddings[:, 1])
        plt.show()

    def get_similar_imgs(self, img, n=10):
        """
        Returns the n most similar images to the given image

        Args:
            img (int, str, Path): index of image in the table, or path to image
            n (int, optional): number of similar images to return. Defaults to 10.

        Returns:
            tuple: (list of paths, list of ids)
        """
        embeddings = None
        if self.table is None:
            LOGGER.error('No embedding space found. Please build the embedding space first.')
            return None
        if isinstance(img, int):
            embeddings = self.table.to_pandas()['vector'][img]
        elif isinstance(img, (str, Path)):
            img_path = img
        else:
            LOGGER.error('img should be index from the table(int) or path of an image (str or Path)')
            return

        if embeddings is None:
            embeddings = self.predictor.embed(img_path).squeeze().cpu().numpy()
        sim = self.table.search(embeddings).limit(n).to_df()
        return sim['path'].to_list(), sim['id'].to_list()

    def plot_similar_imgs(self, img, n=10):
        """
        Plots the n most similar images to the given image

        Args:
            img (int, str, Path): index of image in the table, or path to image.
            n (int, optional): number of similar images to return. Defaults to 10.
        """
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

    def get_similarity_index(self, sim_thres=0.90, top_k=0.01, dim=256, sorted=False):
        """

        Args:
            sim_thres (float, optional): Similarity threshold to set the minimum similarity. Defaults to 0.9.
            top_k (float, optional): Top k fraction of the similar embeddings to apply the threshold on. Defaults to 0.1.
            dim (int, optional): Dimension of the reduced embedding space. Defaults to 256.
            sorted (bool, optional): Sort the embeddings by similarity. Defaults to False.
        Returns:
            np.array: Similarity index
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
        self._sim_index = np.zeros(len(embs))
        limit = max(int(len(embs) * top_k), 1)

        # create a new table with reduced dimensionality to speedup the search
        pca = PCA(n_components=min(dim, len(embs)))
        reduced_embs = pca.fit_transform(embs)
        dim = reduced_embs.shape[1]
        values = pa.array(reduced_embs.reshape(-1), type=pa.float32())
        table_data = pa.FixedSizeListArray.from_arrays(values, dim)
        table = pa.table([table_data, self.table.to_arrow()['id']], names=['vector', 'id'])
        self._reduced_embs_table = self._create_table('reduced_embs', data=table, mode='overwrite')

        # with multiprocessing.Pool() as pool: # multiprocessing doesn't do much. Need to revisit when GIL removal is widely adopted
        #    list(tqdm(pool.imap(build_index, iterable)))

        for _, emb in enumerate(tqdm(reduced_embs)):
            df = self._reduced_embs_table.search(emb).metric('cosine').limit(limit).to_df().query(
                f'score <= {threshold}')
            for idx in df['id'][1:]:
                self._sim_index[idx] += 1
        self._drop_table('reduced_embs')

        return self._sim_index if not sorted else np.sort(self._sim_index)

    def plot_similirity_index(self, sim_thres=0.90, top_k=0.01, dim=256, sorted=False):
        """
        Plots the similarity index

        Args:
            threshold (float, optional): Similarity threshold to set the minimum similarity. Defaults to 0.9.
            top_k (float, optional): Top k fraction of the similar embeddings to apply the threshold on. Defaults to 0.1.
            dim (int, optional): Dimension of the reduced embedding space. Defaults to 256.
            sorted (bool, optional): Whether to sort the index or not. Defaults to False.
        """
        index = self.get_similarity_index(sim_thres, top_k)
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

        self.removed_img_count += len(idxs)

        table = pa_table.filter(mask)
        ids = [i for i in range(len(table))]
        table = table.set_column(1, 'id', [ids])

        # TODO: handle throws error if table is empty
        self.table = self._create_table(self.temp_table_name, data=table, mode='overwrite')  # work on a temporary table

        self.log_status()

    def reset(self):
        """
        Resets the dataset table to its original state or to the last persisted state.
        """
        if self.table is None:
            LOGGER.info('No changes made to the dataset.')
            return

        db = self._connect()
        if self.temp_table_name in db.table_names():
            self._drop_table(self.temp_table_name)

        self.table = self._open_table(self.table_name)
        self.removed_img_count = 0
        # self._sim_index = None # Not sure if we should reset this as computing the index is expensive
        LOGGER.info('Dataset reset to original state.')

    def persist(self, name=None):
        """
        Persists the changes made to the dataset. Available only if data is provided in the constructor.

        Args:
            name (str, optional): Name of the new dataset. Defaults to `data_updated.yaml`.
        """
        db = self._connect()
        if self.table is None or self.temp_table_name not in db.table_names():
            LOGGER.info('No changes made to the dataset.')
            return
        if not self.data:
            LOGGER.info('No dataset provided.')
            return

        LOGGER.info('Persisting changes to the dataset...')
        self.log_status()

        if not name:
            name = self.data.split('.')[0] + '_updated.' + self.data.split('.')[1]
        train_txt = 'train_updated.txt'

        path = Path(self.dataset_info['path']).resolve()  # add new train.txt file in the dataset parent path
        if (path / train_txt).exists():
            (path / train_txt).unlink()  # remove existing

        for img in tqdm(self.table.to_pandas()['path'].to_list()):
            with open(path / train_txt, 'a') as f:
                f.write(f'./{Path(img).relative_to(path).as_posix()}' + '\n')  # add image to txt file

        new_dataset_info = self.dataset_info.copy()
        new_dataset_info['train'] = train_txt
        for key, value in new_dataset_info.items():
            if isinstance(value, Path):
                new_dataset_info[key] = value.as_posix()

        yaml.dump(new_dataset_info, open(Path(self.project) / name, 'w'))  # update dataset.yaml file

        # TODO: not sure if this should be called data_final to prevent overwriting the original data? Creating embs for large datasets is expensive
        self.table = self._create_table(self.table_name, data=self.table.to_arrow(), mode='overwrite')
        db.drop_table(self.temp_table_name)

        LOGGER.info('Changes persisted to the dataset.')
        self._log_training_cmd((Path(self.project) / name).as_posix())

    def log_status(self):
        # TODO: Pretty print log status
        LOGGER.info('\n|-----------------------------------------------|')
        LOGGER.info(f'\t Number of images: {len(self.table.to_arrow())}')
        LOGGER.info(f'\t Number of removed images: {self.removed_img_count}')
        LOGGER.info('|------------------------------------------------|')

    def _log_training_cmd(self, data_path):
        LOGGER.info(
            f'{colorstr("LanceDB: ") }New dataset created successfully! Run the following command to train a model:')
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
        table = db.open_table(name) if name in db.table_names() else None
        if table is None:
            raise ValueError(f'{colorstr("LanceDB: ") }Table not found.')
        return table

    def _drop_table(self, name):
        db = lancedb.connect(self.project)
        if name in db.table_names():
            db.drop_table(name)
            return True

        return False

    def _copy_table_to_project(self, table_path):
        if not table_path.endswith('.lance'):
            raise ValueError(f"{colorstr('LanceDB: ')} Table must be a .lance file")

        LOGGER.info(f'Copying table from {table_path}')
        path = Path(table_path).parent
        name = Path(table_path).stem  # lancedb doesn't need .lance extension
        db = lancedb.connect(path)
        table = db.open_table(name)
        return self._create_table(self.table_name, data=table.to_arrow(), mode='overwrite')

    def _embedding_func(self, imgs):
        embeddings = []
        for img in tqdm(imgs):
            embeddings.append(self.predictor.embed(img, verbose=self.verbose).squeeze().cpu().numpy())
        return embeddings

    def _setup_predictor(self, model, device=''):
        model = YOLO(model)
        predictor = EmbeddingsPredictor(overrides={'device': device})
        predictor.setup_model(model.model)

        return predictor

    def create_index(self):
        # TODO: create index
        pass
