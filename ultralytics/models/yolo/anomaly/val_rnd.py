# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Internal R&D evaluation for YOLO Anomaly: a whole anomaly catalogue, not one dataset.

``val.py`` validates one dataset with one memory bank. This evaluates a MVTec-Ultra tree: a bank
per product built from that product's normal images, every product scored, then the images a tag
query selects pooled into ONE ranked list -- what a single deployed threshold faces.

Not intended for normal users; it exists for internal experiments, and pairs with ``train_rnd.py``
the way ``val.py`` pairs with ``train.py``.
"""

from __future__ import annotations

import math
import random
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from ultralytics.models.yolo.anomaly.val import YOLOAnomalyValidator
from ultralytics.utils import LOGGER, SimpleClass, YAML
from ultralytics.utils.torch_utils import select_device


MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

_MVTEC_ROOT_CANDIDATES = (
    "/data/shared-datasets/louis_data/MVTec-YOLO",
    "/Users/louis/workspace/ultra_louis_work/buffer/AnomalyData/MVTEC/MVTec-YOLO",
    "/home/laughing/codes/datasets/MVTec-YOLO",
)

class GroupMeta:
    """The OOD group taxonomy, loaded from a yaml given by path.

    A *group* is ``<dataset>/<product>/<anomaly_class>`` and is globally unique, so
    ``mvtec/carpet/color`` and ``mvtec/leather/color`` stay distinct despite the colliding class
    name. Each group carries a ``nature`` (``structural`` / ``logical`` / ``excluded``) and each
    product a ``surface``; selecting which groups fitness pools over is a query over those tags
    rather than a list hardcoded here.

    Read **by path, never imported**: the file is generated outside this repo (it encodes reviewed
    human judgement about specific datasets and could never go upstream), so keeping it at arm's
    length is what stops that judgement from becoming a dependency of the training code. The path
    comes from the model yaml's ``anomaly.meta_yaml``, exactly like ``test_root`` already does.
    """

    # Rebuilt datasets flatten the original path into the filename (`test_crack_000.jpg` <-
    # `test/crack/000.png`), so the origin folder is the prefix before the first `_<digits>` group.
    _NAME_RX = re.compile(r"^(.*?)_(\d+)(.*)\.[a-zA-Z]+$")

    SCHEMA = 3  # a mismatch is fatal: a silently empty taxonomy measures nothing and says nothing

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.meta = YAML.load(self.path)
        if (v := self.meta.get("version")) != self.SCHEMA:
            raise ValueError(f"{self.path} is schema v{v}; this code reads v{self.SCHEMA}")
        self.datasets = self.meta.get("datasets", {})
        if not any(d.get("anomalies") for d in self.datasets.values()):
            raise ValueError(f"{self.path} declares no anomaly groups")

    def products(self) -> list[tuple[str, str, str]]:
        """-> (dataset, product, slug) for every product, in file order."""
        return [(ds, p, v["slug"]) for ds, d in self.datasets.items() for p, v in d.get("products", {}).items()]

    def group_of(self, dataset: str, product: str, im_file: str | Path) -> str | None:
        """-> the group id for one evaluated image, or ``None`` if it is a normal image.

        Normal images belong to no anomaly group by construction; they are what the pooled read adds
        back so precision counts false alarms on good product, which a per-group read cannot.
        """
        d = self.datasets.get(dataset) or {}
        m = self._NAME_RX.match(Path(im_file).name)
        if not m:
            return None
        origin = m.group(1)
        if origin in (d.get("good_prefixes") or []):
            return None
        anomaly = re.sub(r"^test_(public_)?", "", origin)
        return f"{dataset}/{product}/{anomaly}"

    def nature_of(self, dataset: str, product: str, im_file: str | Path) -> str | None:
        """-> the nature of ONE evaluated image, or ``None`` if it is normal.

        Nature belongs to the image, not the group: an anomaly folder can hold structural, logical and
        combined images at once (``mvtec/cable/combined`` holds 8 structural and 3 combined). The
        group carries what its images default to and ``image_nature`` lists the exceptions, so the
        yaml stays a few lines instead of one entry per image.
        """
        if not (group := self.group_of(dataset, product, im_file)):
            return None
        d = self.datasets.get(dataset) or {}
        stem = Path(im_file).stem
        if over := (d.get("image_nature") or {}).get(f"{product}/{stem}"):
            return over
        return self.nature(group)

    def deferred(self, dataset: str) -> bool:
        """Is the whole dataset held out of the eval set for now?

        A dataset STATUS, not a nature -- its groups keep their real natures (MVTec-LOCO's 432
        structural images are still labelled structural), so re-admitting it is a one-line change
        rather than a re-labelling job.
        """
        return ((self.datasets.get(dataset) or {}).get("status") or "active") == "deferred"

    def nature(self, group: str) -> str | None:
        ds, _, rest = group.partition("/")
        entry = ((self.datasets.get(ds) or {}).get("anomalies") or {}).get(rest)
        return entry.get("nature") if entry else None

    def surface(self, dataset: str, product: str) -> str | None:
        v = ((self.datasets.get(dataset) or {}).get("products") or {}).get(product)
        return v.get("surface") if v else None

    def selects(self, query: str, dataset: str, product: str, im_file: str | Path) -> bool:
        """Does ``query`` select this one image? The image-level counterpart of :meth:`select`.

        A deferred dataset selects nothing whatever the query says -- that is what deferring means.
        """
        if self.deferred(dataset):
            return False
        query = (query or "").strip()
        tag, _, want = (t.strip() for t in query.partition("="))
        if "=" not in query:
            return self.group_of(dataset, product, im_file) in self.select(query)
        if tag == "nature":
            return self.nature_of(dataset, product, im_file) == want
        if tag == "surface":
            return self.group_of(dataset, product, im_file) is not None and self.surface(dataset, product) == want
        return self.group_of(dataset, product, im_file) in self.select(query)

    def select(self, query: str) -> set[str]:
        """Group ids matching ``<tag>=<value>`` (e.g. ``nature=structural``), or an explicit
        comma-separated list of group ids. Unknown tags raise rather than silently select nothing —
        a fitness query that quietly matches zero groups would train against an empty metric.

        Group-level, so it reads a group's DEFAULT nature. Use :meth:`selects` to decide about an
        individual image; only that one honours ``image_nature`` and the dataset status."""
        query = (query or "").strip()
        if "=" not in query:
            ids = {q.strip() for q in query.split(",") if q.strip()}
            if unknown := {i for i in ids if self.nature(i) is None}:
                raise ValueError(f"unknown group id(s) in fitness_groups: {sorted(unknown)}")
            return ids
        tag, _, want = query.partition("=")
        tag, want = tag.strip(), want.strip()
        out = set()
        for ds, d in self.datasets.items():
            for g, v in (d.get("anomalies") or {}).items():
                product = g.split("/")[0]
                got = v.get(tag) if tag != "surface" else self.surface(ds, product)
                if got == want:
                    out.add(f"{ds}/{g}")
        if not out:
            raise ValueError(f"fitness_groups={query!r} matched no groups in {self.path}")
        return out

class OODResult(SimpleClass):
    """Results of a grouped OOD evaluation over an anomaly catalogue.

    Attributes:
        results_dict (dict[str, float]): Decisive pooled metrics, keys stripped of their pass prefix.
        pooled (dict[str, float]): Pooled metrics for every pass.
        products (list[dict]): One row per product.
        groups (list[dict]): One row per anomaly group. Read these for recall, not precision: a
            group holds only its own anomaly's images, so it has no normal images to false-alarm on.
        decisive (str): Prefix of the pass the decisive metrics come from.
    """

    def __init__(self, pooled: dict, products: list[dict], groups: list[dict], decisive: str):
        """Initialize from the pooled metrics and the diagnostic rows."""
        self.pooled, self.products, self.groups, self.decisive = pooled, products, groups, decisive
        pre = f"pool_{decisive}"
        self.results_dict = {k[len(pre) :]: v for k, v in pooled.items() if k.startswith(pre)}

def _normal_dir_from_yaml(yaml_path: str | Path) -> Path:
    """Resolve the directory of normal images referenced by a data yaml.

    Follows the MVTec convention: if ``<train>/good`` exists, use it; otherwise
    use ``<train>``.
    """
    data = YAML.load(yaml_path)
    root = Path(data.get("path", Path(yaml_path).parent))
    train = Path(data["train"])
    if not train.is_absolute():
        train = root / train
    good = train / "good"
    return good if good.is_dir() else train

@contextmanager
def _frozen_rng():
    """Run a block, then rewind every global RNG it touched.

    Used to keep ``ood_end2end``'s extra eval passes purely ADDITIVE. The memory-bank build
    draws on global RNG and is reproducible only by replaying the same draw sequence: three
    back-to-back ``_run_ood_eval`` calls on one model give prior-ON mAP10_50
    0.1957 / 0.2026 / 0.2019, while prior-OFF (bank disabled) stays at 0.195577 to six
    decimals — so it is the bank, not the detector. Without this, an extra pass would shift
    every later category's prior-ON number and ``ood_end2end=True`` runs would stop being
    comparable with ``False`` ones on the o2m columns they are supposed to share.
    """
    state = (
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        random.getstate(),
        np.random.get_state(),
    )
    try:
        yield
    finally:
        torch.set_rng_state(state[0])
        if state[1] is not None:
            torch.cuda.set_rng_state_all(state[1])
        random.setstate(state[2])
        np.random.set_state(state[3])

def _average_ood_rows(rows: list[dict]) -> dict[str, float]:
    """Macro-average over categories, ignoring NaNs and non-numeric fields (e.g. ``category``).

    Averages every numeric key present, so both the heatmap-prior metrics (``mAP50`` …) and the
    none-prior metrics (``none_mAP50`` …) are aggregated in one pass.
    """
    keys = [k for k in rows[0] if isinstance(rows[0].get(k), (int, float))]
    out: dict[str, float] = {}
    for key in keys:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and not math.isnan(r[key])]
        out[key] = sum(vals) / len(vals) if vals else math.nan
    return out

class OODEvaluator:
    """Evaluate a model against an anomaly catalogue: per-product, per-group, and pooled.

    Attributes:
        cfg (dict): Resolved catalogue config -- ``test_root``, ``meta_yaml``, ``test_batch``.
        query (str): Which anomalies the pooled number is over, e.g. ``nature=structural``.
        e2e (bool): Also run the o2o (NMS-free) passes, which carry the decisive metrics.
        passes (set[str]): Which of the four passes to run (``""`` heatmap, ``"none_"``,
            ``"e2e_"``, ``"e2e_none_"``). Prior-OFF-only subsets skip the memory-bank build
            entirely -- a bank is never consulted on those passes, so building one is pure cost.

    Examples:
        >>> ev = OODEvaluator("/data/.../MVTec-Ultra/v1")
        >>> res = ev(model)
        >>> res["mAP50@0.25"]
    """

    _PASSES = ("", "none_", "e2e_", "e2e_none_")

    def __init__(
        self,
        data: str | Path | dict,
        groups: str = "nature=structural",
        e2e: bool = True,
        passes: str | list[str] | None = None,
        device=None,
        batch: int = 8,
        workers: int = 8,
        save_dir: str | Path | None = None,
        verbose: bool = False,
    ):
        """Initialize from a MVTec-Ultra version root, or from a model yaml's ``anomaly`` config.

        Args:
            data (str | Path | dict): A catalogue root whose ``meta.yaml`` is the taxonomy, or the
                model yaml's ``anomaly`` dict for the legacy ``test_root`` / ``test_categories``
                form.
            groups (str): Tag query (``nature=structural``, ``surface=texture``) or a
                comma-separated list of group ids.
            e2e (bool): Run the o2o passes as well as o2m.
            passes (str | list[str], optional): Subset of ``("", "none_", "e2e_", "e2e_none_")``
                to run; ``None`` runs all four. A subset without a prior-ON pass skips the
                memory-bank build.
            device (str | int | torch.device, optional): Defaults to the auto-selected device.
            batch (int): Batch size for bank building and validation.
            workers (int): Dataloader workers.
            save_dir (str | Path, optional): Write ``ood_percat.csv`` / ``ood_pergroup.csv`` /
                ``pooled.csv`` here.
            verbose (bool): Print each product's validation table.
        """
        if isinstance(data, dict):
            self.cfg = dict(data)
        else:
            data = Path(data)
            self.cfg = {"test_root": str(data), "meta_yaml": str(data / "meta.yaml"), "test_batch": batch}
        self.cfg.setdefault("test_batch", batch)
        self.query, self.e2e, self.verbose = groups, e2e, verbose
        if not passes:
            # None, "" and [] all mean "no pruning" — an empty subset would run nothing and
            # produce an empty result silently.
            self._want = set(self._PASSES)
        else:
            self._want = {passes} if isinstance(passes, str) else set(passes)
            if unknown := self._want - set(self._PASSES):
                raise ValueError(f"unknown pass(es) {sorted(unknown)}; expected a subset of {self._PASSES!r}")
        self.device = select_device(device) if device is None else device
        self.workers, self.epoch = workers, -1
        self.save_dir = Path(save_dir) if save_dir else None
        self.meta = GroupMeta(self.cfg["meta_yaml"]) if self.cfg.get("meta_yaml") else None

    def _decisive(self) -> str:
        """The pass prefix the decisive metrics come from: the strongest pass actually run.

        Matches the old rule (``e2e_none_`` when e2e, else ``none_``) whenever those passes are in
        the requested set; otherwise steps down to whatever pass was requested.
        """
        for pre in self._PASSES[::-1]:
            if pre in self._want and (self.e2e or not pre.startswith("e2e_")):
                return pre
        return ""

    def __call__(self, model, epoch: int = -1) -> OODResult:
        """Run the full evaluation.

        Args:
            model (YOLOAnomalyModel): A raw model, already on the target device.
            epoch (int): Recorded in the csv rows; -1 means an offline run.

        Returns:
            (OODResult): Decisive pooled metrics, with the per-product and per-group rows.

        Raises:
            RuntimeError: If any product failed. A partial aggregate is a different measurement.
        """
        self.epoch = epoch
        yamls = self.resolve()
        if not yamls:
            raise ValueError(f"no product yamls under {self.cfg.get('test_root')}")

        products, passes = self.run(model, yamls)
        if len(products) != len(yamls):
            raise RuntimeError(f"OOD eval incomplete: {len(products)}/{len(yamls)} products succeeded")

        groups = self.group_rows(passes) if self.meta else []
        pooled = self.pooled(passes) if self.meta else {}
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.save_rows(products, "ood_percat.csv", "category")
            if groups:
                self.save_rows(groups, "ood_pergroup.csv", "group")
            (self.save_dir / "pooled.csv").write_text(
                "key,value\n" + "".join(f"{k},{v:.6g}\n" for k, v in sorted(pooled.items()))
            )
        return OODResult(pooled, products, groups, self._decisive())

    def resolve(self) -> list[tuple[str, str, Path]]:
        """Resolve the OOD products as ``(dataset, product, yaml)`` triples.

        ``dataset``/``product`` are what attributes an evaluated image to its group, so they travel
        with the yaml rather than being re-derived from the path later -- the directory slug encodes
        them (``mvtec-metal-nut-osplit``) only through a substitution that would silently stop
        matching. Legacy configs keep working as the single ``mvtec`` dataset.

        ``test_skip_unselected`` drops products holding no selected group. It cannot change the
        pooled number -- such a product contributes no images, normals included -- but the skipped
        products lose their diagnostic rows, so it is opt-in.
        """
        if meta_path := self.cfg.get("meta_yaml"):
            root = Path(self.cfg.get("test_root") or ".")
            meta = self.meta or GroupMeta(meta_path)
            # An empty query selects nothing, so skip_unselected would drop every product.
            selected = self.meta.select(self.query) if self.cfg.get("test_skip_unselected") and (self.query or "").strip() else None
            out = []
            for ds, product, slug in meta.products():
                if selected is not None:
                    # Group-level check: an image_nature override could still deselect every image
                    # of a kept product, and pooled() then drops it with its normals -- the skip
                    # here is an optimization of exactly that outcome, so the pooled number is
                    # unchanged by construction.
                    anomalies = (meta.datasets.get(ds) or {}).get("anomalies") or {}
                    if not any(f"{ds}/{g}" in selected for g in anomalies if g.startswith(f"{product}/")):
                        LOGGER.debug(f"OOD eval: skipping {ds}/{product} (no selected group)")
                        continue
                d = root / slug
                for name in (f"{slug}_binary.yaml", "data.yaml"):
                    if (p := d / name).exists():
                        out.append((ds, product, p))
                        break
                else:
                    LOGGER.warning(f"no data yaml for {ds}/{product} under {d}")
            return out

        if explicit := self.cfg.get("test_data_yamls"):
            return [("mvtec", Path(p).parent.name, Path(p)) for p in explicit]

        root = self.cfg.get("test_root")
        if not root:
            for candidate in _MVTEC_ROOT_CANDIDATES:
                if Path(candidate).is_dir():
                    root = candidate
                    break
        if not root:
            LOGGER.warning("AnomalyRNDTrainer: no test_data_yamls or test_root configured; skipping OOD eval.")
            return []

        cats = self.cfg.get("test_categories") or MVTEC_CATEGORIES
        yamls = []
        for cat in cats:
            for name in (f"{cat}_binary.yaml", f"{cat}.yaml"):
                p = Path(root) / cat / name
                if p.exists():
                    yamls.append(("mvtec", cat, p))
                    break
            else:
                LOGGER.warning(f"AnomalyRNDTrainer: no yaml found for category '{cat}' under {root}")
        return yamls

    def run(self, model, yamls: list[tuple[str, str, Path]]) -> tuple[list[dict], dict]:
        """Fit bank per product and validate.

        Returns the per-category rows **and** the finished validators, keyed by the prefix their
        metrics carry (``""`` o2m/prior-on, ``"none_"``, ``"e2e_"``, ``"e2e_none_"``). Keeping them
        is what makes the grouped and pooled reads free: each still holds its pass's flat stat
        arrays plus the image order, so any image subset re-scores without a second inference pass.
        """
        rows = []
        failed: list[str] = []
        passes: dict[str, list[tuple[str, str, YOLOAnomalyValidator]]] = {}
        batch = int(self.cfg.get("test_batch", 8))
        device = self.device
        workers = self.workers

        want, e2e = self._want, bool(self.e2e)
        # Prior-OFF passes never consult the bank (its forward returns zeros while `building`),
        # so a subset without a prior-ON pass skips the build entirely — pure eval-time cost.
        need_bank = ("" in want) or ("e2e_" in want)
        for dataset, product, yaml in yamls:
            source = _normal_dir_from_yaml(yaml)
            try:
                model.memory_bank.reset()
                if need_bank:
                    n = model.build_memory_bank(str(source), imgsz=640, device=device, batch=batch)
                    if not n:
                        LOGGER.warning(f"OOD eval: empty bank for {yaml.name}; skipping.")
                        continue

                overrides = {
                    "task": "detect",
                    "mode": "val",
                    "data": str(yaml),
                    # The official test split is declared as `val:`; see the dataset's README.
                    "split": self.cfg.get("test_split", "val"),
                    "imgsz": 640,
                    "batch": batch,
                    "workers": workers,
                    "device": str(device) if device is not None else None,
                    "rect": False,
                    "plots": False,
                    "verbose": False,
                    "save_json": False,
                    "single_cls": True,
                    "iou": 0.2,
                    # AP is threshold-free; a 0.25 floor here deleted ~73% of correct detections
                    # before scoring. The 0.25 numbers are re-derived by masking.
                    "conf": 0.001,
                    "end2end": False,
                }
                keep = passes.setdefault
                row = {"category": yaml.parent.name}

                # Pass 1: heatmap prior (memory bank active) — the yoloa_clean fitness signal.
                if "" in want:
                    validator = YOLOAnomalyValidator(args=overrides)
                    validator(trainer=None, model=model)
                    row.update(validator._ood_map_metrics())
                    keep("", []).append((dataset, product, validator.snapshot()))

                # Pass 2: prior OFF (bank disabled via `building`) — the bare-detector baseline.
                if "none_" in want:
                    mb = model.memory_bank
                    saved_building = mb.building
                    mb.building = True
                    try:
                        validator_none = YOLOAnomalyValidator(args=overrides)
                        validator_none(trainer=None, model=model)
                        row.update({f"none_{k}": v for k, v in validator_none._ood_map_metrics().items()})
                        keep("none_", []).append((dataset, product, validator_none.snapshot()))
                    finally:
                        mb.building = saved_building

                # Passes 3-4: the same two on the o2o branch — the NMS-free path a deployment ships.
                if e2e and (("e2e_" in want) or ("e2e_none_" in want)):
                    e2e_overrides = {**overrides, "end2end": True}
                    with _frozen_rng():  # keeps the o2m columns identical to an ood_end2end=False run
                        if "e2e_" in want:
                            v_e2e = YOLOAnomalyValidator(args=e2e_overrides)
                            v_e2e(trainer=None, model=model)
                            row.update({f"e2e_{k}": v for k, v in v_e2e._ood_map_metrics().items()})
                            keep("e2e_", []).append((dataset, product, v_e2e.snapshot()))

                        if "e2e_none_" in want:
                            mb = model.memory_bank
                            saved_building = mb.building
                            mb.building = True
                            try:
                                v_e2e_none = YOLOAnomalyValidator(args=e2e_overrides)
                                v_e2e_none(trainer=None, model=model)
                                row.update({f"e2e_none_{k}": v for k, v in v_e2e_none._ood_map_metrics().items()})
                                keep("e2e_none_", []).append((dataset, product, v_e2e_none.snapshot()))
                            finally:
                                mb.building = saved_building

                rows.append(row)
            except Exception as e:
                failed.append(f"{dataset}/{product}: {type(e).__name__}: {e}")
                LOGGER.warning(f"OOD eval failed for {yaml}: {type(e).__name__}: {e}")

        if failed:
            LOGGER.error(f"OOD eval INCOMPLETE: {len(rows)}/{len(yamls)} products; failed: {failed}")
        return rows, passes

    @staticmethod
    def _group_index(meta: GroupMeta, validator, dataset: str, product: str) -> tuple[dict, list]:
        """Split one finished pass's images into ``{group_id: [index]}`` and the normal ones.

        Normal images carry no anomaly group by construction. They are held separately because the
        two reads need them differently: a per-group row must NOT contain them (it would then
        measure two things), while the pooled read must, or precision never counts a false alarm on
        good product -- which is most of what a deployed detector gets wrong.
        """
        groups: dict[str, list[int]] = {}
        good: list[int] = []
        for i, f in enumerate(validator._ood_files):
            if g := meta.group_of(dataset, product, f):
                groups.setdefault(g, []).append(i)
            else:
                good.append(i)
        return groups, good

    def group_rows(self, passes: dict) -> list[dict]:
        """One diagnostic row per group per epoch, all four passes merged into the row.

        **Read these for recall.** A group holds only its own anomaly's images, so its ``P`` is
        "false detections inside this anomaly's images" -- not a deployment precision, which needs
        the normal images the pooled read supplies. Reported anyway because a group whose precision
        collapses while recall holds is a real signal, just not a shippable number.
        """
        rows: dict[str, dict] = {}
        for pre, entries in passes.items():
            for dataset, product, v in entries:
                for g, idx in self._group_index(self.meta, v, dataset, product)[0].items():
                    # The group's nature is its images' default; a group whose images disagree is
                    # reported as `<default>+<other>` rather than silently as the majority.
                    nats = {self.meta.nature_of(dataset, product, v._ood_files[i]) for i in idx}
                    base = self.meta.nature(g) or "unknown"
                    r = rows.setdefault(
                        g,
                        {
                            "group": g,
                            "nature": base if nats <= {base} else "+".join(sorted(n or "?" for n in nats)),
                            "surface": self.meta.surface(dataset, product) or "unknown",
                            "status": "deferred" if self.meta.deferred(dataset) else "active",
                            "n": len(idx),
                        },
                    )
                    r.update({f"{pre}{k}": val for k, val in v._ood_map_metrics(images=idx).items()})
        return [rows[g] for g in sorted(rows)]

    def pooled(self, passes: dict) -> dict[str, float]:
        """Pooled binary AP over the groups ``query`` selects — the decisive number and fitness.

        Selection set = the selected anomalous images ∪ every normal image of the products that
        contributed at least one. A product contributing nothing selected stays out entirely,
        normals included: its good images would otherwise add false-alarm opportunities for an anomaly
        the selection never asked about.

        One ``ap_per_class`` over the concatenated ranked list, not a mean of per-product APs.
        ``single_cls=True`` makes every class 0, which is what lets four datasets pool at all.
        """
        out: dict[str, float] = {}
        for pre, entries in passes.items():
            vs, sel, off = [], [], 0
            for dataset, product, v in entries:
                good = self._group_index(self.meta, v, dataset, product)[1]
                # Image by image, not group by group: nature belongs to the image, so a group can
                # contribute some of its images and not others.
                hit = [i for i, f in enumerate(v._ood_files) if self.meta.selects(self.query, dataset, product, f)]
                if not hit:
                    continue
                vs.append(v)
                sel += [off + i for i in hit + good]
                off += len(v._ood_files)
            if vs:
                pool = YOLOAnomalyValidator.pooled(vs)
                out.update({f"pool_{pre}{k}": val for k, val in pool._ood_map_metrics(images=sel).items()})
        return out

    def save_rows(self, rows: list[dict], fname: str, key: str) -> None:
        """Append per-category (``ood_percat.csv``) or per-group (``ood_pergroup.csv``) OOD rows.

        ``_average_ood_rows`` collapses 15 categories into the single ``ood/*`` number that goes into
        results.csv, and the rows it averaged were then dropped — but that average hides a 2.6x
        texture/object spread, and a checkpoint chosen on it can halve one category's recall while the
        mean moves 0.0020. The rows already exist, so keeping them (plus ``save_period`` weights) makes
        checkpoint selection re-decidable offline under a different aggregate, with no extra eval.

        Long format — ``epoch, category, <metric>…`` — so the file stays readable when OOD eval is
        gated by ``test_val_freq`` and only some epochs have rows.
        """
        keys = [k for k in rows[0] if k != key]
        csv = self.save_dir / fname
        header = "" if csv.exists() else f"epoch,{key}," + ",".join(keys) + "\n"

        def cell(v):  # group rows carry `nature` / `surface` alongside the numbers
            return f"{v:.6g}" if isinstance(v, (int, float)) else str(v)

        with open(csv, "a", encoding="utf-8") as f:
            f.write(header)
            f.writelines(
                f"{self.epoch + 1},{r[key]}," + ",".join(cell(r.get(k, math.nan)) for k in keys) + "\n"
                for r in rows
            )

