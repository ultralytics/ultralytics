"""Recompute the pooled OOD metrics offline, from a dump written by val_yoloa_mvtec_ultra.py --dump.

The point of this file is that it computes NOTHING itself. Matching is ultralytics'
``match_predictions``, scoring is ``YOLOAnomalyValidator._ood_map_metrics``, and the selection is
``GroupMeta`` -- the same three pieces the in-loop path uses. A number that differs from the run
that produced the dump is therefore a bug here, never a second opinion.

    PYTHONPATH=. python tools/analyze_ood_dump.py preds.jsonl --data <MVTec-Ultra root>

Every metric is reported BOTH ways. The bare column is micro -- one ranked list over that row's
images, scored once; the ``macro_`` column gives each of the row's groups one vote. They answer
different questions and disagree several-fold, so neither is chosen for the reader.

`scope` says which axis a row PINS, never how it was computed: `all` pins none, `group` pins one
group.

Five scopes, differing only in which images they select:

    all       the --groups query                      normals of contributing products included
    dataset   the --groups query, one dataset at a time                                  included
    nature    nature=<value>, its own query (see below)                                  included
    group     one anomaly group's own images                                    EXCLUDED, see below

``nature`` rows ignore ``--groups`` on purpose: under the default ``nature=structural`` an intersect
would leave exactly one nature row, which is not a breakdown. ``group`` rows exclude normal images
because a group holds only its own anomaly's images -- their ``P`` is "false detections inside this
anomaly's images", not a deployment precision, exactly as in ``OODEvaluator.group_rows``.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ultralytics.models.yolo.anomaly.val import YOLOAnomalyValidator
from ultralytics.models.yolo.anomaly.val_rnd import GroupMeta, OODEvaluator

DECISIVE = ("mAP50", "mAP50@0.25", "R50@0.25", "P50@0.25")
TAGS = ("group", "nature", "surface", "status")
COUNTS = ("n", "n_defect", "instances")


def load(path: str) -> YOLOAnomalyValidator:
    """Replay a dump into a read-only validator holding the same flat stats the in-loop pass built."""
    v = YOLOAnomalyValidator.__new__(YOLOAnomalyValidator)
    v.iouv, v.names = torch.linspace(0.1, 0.5, 9), {0: "anomaly"}
    v.niou = v.iouv.numel()

    files, tp, conf, pcls, tcls, pimg, gimg = [], [], [], [], [], [], []
    for i, line in enumerate(open(path, encoding="utf-8")):
        r = json.loads(line)
        files.append(r["file"])
        p = {"bboxes": torch.tensor(r["bboxes"], dtype=torch.float32).reshape(-1, 4),
             "cls": torch.tensor(r["cls"], dtype=torch.float32)}
        g = {"bboxes": torch.tensor(r["gt_bboxes"], dtype=torch.float32).reshape(-1, 4),
             "cls": torch.tensor(r["gt_cls"], dtype=torch.float32)}
        tp.append(v._process_batch(p, g)["tp"])  # ultralytics' matcher, not a copy of it
        conf.append(np.asarray(r["conf"], dtype=np.float32))
        pcls.append(np.asarray(r["cls"], dtype=np.float32))
        tcls.append(np.asarray(r["gt_cls"], dtype=np.float32))
        pimg.append(np.full(len(r["conf"]), i, dtype=np.int64))
        gimg.append(np.full(len(r["gt_cls"]), i, dtype=np.int64))

    v._ood_files = files
    v._ood_stats = {"tp": np.concatenate(tp), "conf": np.concatenate(conf),
                    "pred_cls": np.concatenate(pcls), "target_cls": np.concatenate(tcls)}
    v._ood_img = {"pred": np.concatenate(pimg), "gt": np.concatenate(gimg)}
    return v


def blocks(v: YOLOAnomalyValidator, meta: GroupMeta) -> list[tuple[str, str, list[str], list[int]]]:
    """Split the flat dump back into per-product blocks: ``(dataset, product, files, dump indices)``.

    The dump is one flat file in evaluation order; every selection rule below is per-product, so
    this is the one place that has to recover which product an image came from.
    """
    slugs = {slug: (ds, product) for ds, product, slug in meta.products()}
    per: dict[str, tuple[list[str], list[int]]] = {}
    for i, f in enumerate(v._ood_files):
        slug = next((p for p in f.split("/") if p in slugs), None)
        if slug is None:
            raise ValueError(f"cannot attribute {f} to a product under the given --data")
        files, where = per.setdefault(slug, ([], []))
        files.append(f)
        where.append(i)
    return [(*slugs[k], *per[k]) for k in per]


def pooled_idx(meta: GroupMeta, query: str, blks: list) -> list[int]:
    """Dump indices ``query`` pools over, across ``blks``.

    The rule for which images a query keeps is ``OODEvaluator.select_images`` and is not restated
    here -- a second copy of it would be the second ruler this whole file exists to avoid. All this
    adds is mapping that method's block-local indices back into dump order.
    """
    kept = OODEvaluator.select_images(meta, query, [(ds, p, files) for ds, p, files, _ in blks])
    return sorted(blks[b][3][i] for b, idx in kept for i in idx)


def rollups(v: YOLOAnomalyValidator, meta: GroupMeta, query: str, blks: list):
    """Yield ``(scope, dump indices, tags)`` for every row, in report order.

    The axes a row is sliced on are ``tags`` -- real columns, so the grid can filter and sort on
    them, never parsed back out of a label; a tag left unset spans every value of that axis.

    ``group`` is the taxonomy PATH the row pins, and a shorter path is a broader row: ``mvtec`` is
    a whole dataset, ``mvtec/cable/combined`` one anomaly group. One column rather than two,
    because a group id already starts with its dataset -- filtering on ``mvtec-3d`` reaches both.
    See the module docstring for what each scope selects.
    """
    datasets = sorted({b[0] for b in blks})
    of = lambda ds: [b for b in blks if b[0] == ds]  # noqa: E731 -- the blocks of one dataset
    # A tag query pins an axis; a bare list of group ids pins none, and only `query` records it.
    tag, _, want = (t.strip() for t in (query or "").partition("="))
    pinned = {tag: want} if want and tag in TAGS else {}

    yield "all", pooled_idx(meta, query, blks), pinned  # pins no axis; NOT "the pooled one"

    for ds in datasets:
        if idx := pooled_idx(meta, query, of(ds)):
            yield "dataset", idx, {"group": ds, **pinned}

    natures = {meta.nature_of(ds, p, f) for ds, p, files, _ in blks for f in files}
    for nat in sorted(n for n in natures if n):
        q = f"nature={nat}"
        if idx := pooled_idx(meta, q, blks):
            yield "nature", idx, {"nature": nat}
        # The cross, because the two axes above each hide the other: `nature=logical` pools four
        # datasets into one number, and the dataset rows are all at the --groups nature only.
        for ds in datasets:
            if idx := pooled_idx(meta, q, of(ds)):
                yield "dataset_nature", idx, {"group": ds, "nature": nat}

    groups: dict[str, tuple[list[int], dict, set]] = {}
    for ds, product, files, where in blks:
        for f, i in zip(files, where):
            if g := meta.group_of(ds, product, f):  # None = a normal image, which no group owns
                # `status` is load-bearing, not decoration: a deferred dataset still gets group rows
                # (as in OODEvaluator.group_rows) but contributes nothing to any row above, so
                # without this column its groups read as if they were part of the pooled number.
                idx, tags, nats = groups.setdefault(g, ([], {
                    "nature": meta.nature(g) or "unknown",
                    "surface": meta.surface(ds, product) or "unknown",
                    "status": "deferred" if meta.deferred(ds) else "active",
                }, set()))
                idx.append(i)
                nats.add(meta.nature_of(ds, product, f))
    for g in sorted(groups):
        idx, tags, nats = groups[g]
        tags["group"] = g
        # Nature belongs to the image, so a group's images can disagree with its default. Reported
        # as `a+b` rather than silently as the majority -- the rule OODEvaluator.group_rows uses,
        # and these rows have to match that file's to the digit.
        if not nats <= {tags["nature"]}:
            tags["nature"] = "+".join(sorted(n or "?" for n in nats))
        yield "group", idx, tags


def per_group(v: YOLOAnomalyValidator, meta: GroupMeta, blks: list, idx) -> dict[str, list[int]]:
    """Split a row's images into the anomaly groups they belong to; normal images belong to none."""
    want = set(idx)
    out: dict[str, list[int]] = {}
    for ds, product, files, where in blks:
        for f, i in zip(files, where):
            if i in want and (g := meta.group_of(ds, product, f)):
                out.setdefault(g, []).append(i)
    return out


def macro(v: YOLOAnomalyValidator, per: dict[str, list[int]]) -> dict[str, float]:
    """Unweighted mean of the per-group metrics -- the OTHER aggregation, next to the micro one.

    Micro pools every image into one ranked list, so a big group dominates and a single threshold
    faces exactly that number. Macro gives every group one vote, so a product with 5 images counts
    as much as one with 90. They answer different questions and on this data differ several-fold,
    which is why both are reported rather than one being chosen.

    Composition differs too, unavoidably: the groups hold only anomalous images, while the micro
    row also carries the normal images false alarms are counted on. Macro precision is therefore
    not a deployment precision -- the same caveat OODEvaluator.group_rows carries.

    Each group is scored over THIS row's images, not over the group's full image set, so a group
    whose images split across natures contributes a different number here than its own group row
    shows. Averaging the visible group rows by hand therefore lands close but not exact -- on the
    catalogue that is one group in 117 (`mvtec/cable/combined`, 8 of its 11 images are structural).
    """
    if not per:
        return {}
    rows = [v._ood_map_metrics(images=i) for i in per.values()]
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}


def main() -> None:
    """Replay a dump and report every scope, micro and macro, for one group query."""
    ap = argparse.ArgumentParser(description="Recompute OOD metrics from a prediction dump, sliced four ways.")
    ap.add_argument("dump", help="jsonl written by val_yoloa_mvtec_ultra.py --dump")
    ap.add_argument("--data", required=True, help="the MVTec-Ultra root whose meta.yaml is the taxonomy")
    ap.add_argument("--groups", default="nature=structural", help="tag query, or a list of group ids")
    ap.add_argument("--csv", help="write every scope and all 16 metrics here (long format)")
    ap.add_argument("--md", help="write the non-group scopes here as a markdown table, to paste into run.md")
    a = ap.parse_args()

    v = load(a.dump)
    meta = GroupMeta(f"{a.data}/meta.yaml")
    blks = blocks(v, meta)
    rows, seen = [], set()
    for scope, idx, tags in rollups(v, meta, a.groups, blks):
        # Two scopes can land on the identical image set -- `all` under a nature query IS the
        # `nature` row for it, and a dataset row IS its cross cell. Keep the first, which is the
        # more general scope, and drop the restatement: a duplicate row is not a second reading.
        sel = tuple(idx)
        if scope != "group" and sel in seen:
            continue
        seen.add(sel)
        counts = dict(zip(COUNTS, OODEvaluator._counts(v, idx)))
        tags = {k: tags.get(k, "-") for k in TAGS}
        # 4 decimals, uniformly: the house rule for every reported metric, and it also keeps the
        # csv narrow enough to read aligned -- full float repr padded every metric column to 23.
        fmt = lambda d: {k: f"{val:.4f}" for k, val in d.items()}  # noqa: E731
        per = per_group(v, meta, blks, idx)
        rows.append({"scope": scope, **tags, **counts, "n_groups": len(per) or "-",
                     **fmt(v._ood_map_metrics(images=idx)),
                     **{f"macro_{k}": val for k, val in fmt(macro(v, per)).items()}})

    print(f"{len(v._ood_files)} images in dump · {len(blks)} products · "
          f"{len(v._ood_stats['conf'])} predictions · groups={a.groups!r}")
    # Each cell is micro / macro -- the two aggregations side by side, since choosing one of
    # them for the reader is exactly what hides a several-fold disagreement.
    head = (f"\n{'scope':<15}{'group':<26}{'nature':<12}{'n':>5}{'def':>6}{'grp':>5}  "
            + "  ".join(f"{k + ' mi/ma':>17}" for k in DECISIVE))
    print(head + "\n" + "-" * len(head.strip()))
    for r in rows:
        if r["scope"] == "group":  # 131 of these; they go to the csv, not the terminal
            continue
        print(f"{r['scope']:<15}{r['group']:<26}{r['nature']:<12}"
              f"{r['n']:>5}{r['n_defect']:>6}{r['n_groups']:>5}  "
              + "  ".join(f"{r[k] + ' / ' + r['macro_' + k]:>17}" for k in DECISIVE))

    n_group = sum(r["scope"] == "group" for r in rows)
    if a.md:
        # Absolutes only, no Δ column: one dump is one run, and §6.1 forbids inventing a reference
        # to fill a delta line. Comparing two runs is a different artifact and a different tool.
        top = [r for r in rows if r["scope"] != "group"]
        cols = ["scope", "group", "nature", *COUNTS, "n_groups", *DECISIVE, *(f"macro_{k}" for k in DECISIVE)]
        Path(a.md).write_text(
            "Single-pass dump (the records carry no pass tag -- name it from the run that wrote them).\n"
            "`scope` says which axis a row PINS, not how it was computed; `group` is the taxonomy path it\n"
            "pins, and a shorter path is a broader row. `n` includes the normal images of contributing\n"
            "products; `n_defect` is the images carrying a GT box; `n_groups` is what macro averages.\n"
            "Two aggregations per metric. The bare column is MICRO: every image in one ranked list, so a\n"
            "big group dominates and this is what one deployed threshold faces. `macro_` gives every\n"
            "group one vote instead, over anomalous images only -- so its precision counts no false\n"
            "alarm on good product. They disagree several-fold here; neither is the right one.\n"
            "Absolutes only -- a single run has no reference, so there is no delta column.\n"
            "A scope with no rows under it selected the same images as a row already above it\n"
            "(e.g. a nature carried by one dataset), not nothing.\n\n"
            + "| " + " | ".join(cols) + " |\n| " + " | ".join("---" for _ in cols) + " |\n"
            + "".join("| " + " | ".join(str(r[c]) for c in cols) + " |\n" for r in top),
            encoding="utf-8",
        )
        print(f"\n-> {a.md}  ({len(top)} rows, group rows excluded)")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        n_met = len(rows[0]) - len(TAGS) - len(COUNTS) - 1
        print(f"\n-> {a.csv}  ({len(rows)} rows incl. {n_group} groups × {n_met} metrics)")
    else:
        print(f"\n{n_group} group rows computed but not shown; pass --csv to keep them")


if __name__ == "__main__":
    main()
