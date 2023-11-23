# FastReID Demo

We provide a command line tool to run a simple demo of builtin models.

You can run this command to get cosine similarites between different images

```bash
python demo/visualize_result.py --config-file logs/dukemtmc/mgn_R50-ibn/config.yaml \
--parallel --vis-label --dataset-name DukeMTMC --output logs/mgn_duke_vis \
--opts MODEL.WEIGHTS logs/dukemtmc/mgn_R50-ibn/model_final.pth
```
