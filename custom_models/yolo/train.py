import argparse
import os.path
from pathlib import Path
import yaml
import torch
from model import YOLOV5m
from loss import YOLO_LOSS
from ultralytics_loss import ComputeLoss
from torch.optim import Adam

from utils.validation_utils import YOLO_EVAL
from utils.training_utils import train_loop, get_loaders
from utils.utils import save_checkpoint, load_model_checkpoint, load_optim_checkpoint
from utils.plot_utils import save_predictions
import config
import spdlog as spd

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="coco", help="Path to dataset")
    parser.add_argument("--box_format", type=str, default="yolo", help="Choose between 'coco' and 'yolo' format")
    parser.add_argument("--nosaveimgs", action='store_true', help="Don't save images predictions in SAVED_IMAGES folder")
    parser.add_argument("--nosavemodel", action='store_true', help="Don't save model weights in SAVED_CHECKPOINT folder")
    parser.add_argument("--epochs", type=int, default=100, help="Num of training epochs")
    parser.add_argument("--ultralytics_loss", action='store_true', help="Uses ultralytics loss function")
    parser.add_argument("--nosavelogs", action='store_true', help="Don't save train and eval logs on train_eval_metrics")
    parser.add_argument("--rect", action='store_true', help="Performs rectangular training")
    parser.add_argument("--batch_size", type=int, default=16, help="Set dataloaders batch_size")
    parser.add_argument("--workers", type=int, default=4, help="Set number of workers")
    parser.add_argument("--resume", action='store_true', help="Resume training on a saved checkpoint")
    parser.add_argument("--filename", type=str, help="Model name to use for resume training")
    parser.add_argument("--load_coco_weights", action='store_true', help="Loads Ultralytics weights, (~273 epochs on MS COCO)")
    parser.add_argument("--only_eval", action='store_true', help="Performs only the evaluation (no training loop")

    return parser.parse_args()


def main(opt):
    parent_dir = os.path.abspath(Path(__file__))
    parent_dir = "/".join(parent_dir.split("/")[:-2])
    
    ROOT_DIR = os.path.join(parent_dir, "datasets", opt.data)

    # logger = spd.FileLogger(__file__, './logs.log')
    # logger.set_level(spd.LogLevel.DEBUG)
    # logger.info('Hello')
    # logger.debug('debug')

    if os.path.isfile(os.path.join(ROOT_DIR, "defect.yaml")):
        with open(os.path.join(ROOT_DIR, "defect.yaml"), "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            nc = data["nc"]
            labels = data["names"]

    else:
        assert config.nc is not None and config.labels is not None, "set in config.py nc=num_classes and config.labels='your labels'"

        nc = config.nc
        labels = config.labels

    first_out = config.FIRST_OUT
    scaler = torch.cuda.amp.GradScaler()

    model = YOLOV5m(first_out=first_out, nc=nc, anchors=config.ANCHORS,
                    ch=(first_out * 4, first_out * 8, first_out * 16), inference=False).to(config.DEVICE)

    optim = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # if no models are saved in checkpoints, creates model0 files,
    # else i.e. if model0.pt is in the folder, new filename will be model1.pt
    starting_epoch = 1
    # if loading pre-existing weights

    if opt.load_coco_weights:
        # if dataset is coco loads all the weights
        if opt.data == "coco":
            model.load_state_dict(torch.load("./yolov8n.pt.pt"), strict=True)
            # loads all coco weights but the heads
        else:
            model.load_state_dict(torch.load("./yolov8n.pt.pt"), strict=False)
        
    if "model" not in "".join(os.listdir("./SAVED_CHECKPOINT")):
        filename = "model_1"
        
    elif opt.resume:
        filename = opt.filename
        folder = os.listdir(os.path.join("./SAVED_CHECKPOINT", opt.filename))
        last_epoch = max([int(ckpt.split(".")[0].split("_")[-1]) for ckpt in folder])
        
        load_model_checkpoint(opt.filename, model, last_epoch)
        load_optim_checkpoint(opt.filename, optim, last_epoch)
        starting_epoch = last_epoch + 1
    
    else:
        models_saved = os.listdir("./SAVED_CHECKPOINT")
        models_saved = [int(model_name.split("_")[1]) for model_name in models_saved if "model" in model_name] # gets rid of weird files

        filename = "model_" + str(max(models_saved)+1)

    save_logs = False if opt.nosavelogs else True
    rect_training = True if opt.rect else False

    # check get_loaders to see how augmentation is set
    train_loader, val_loader = get_loaders(db_root_dir=ROOT_DIR, batch_size=opt.batch_size, num_classes=nc,
                                           box_format=opt.box_format, ultralytics_loss=opt.ultralytics_loss,
                                           rect_training=rect_training, num_workers=opt.workers)

    if opt.ultralytics_loss:
        loss_fn = ComputeLoss(model, save_logs=save_logs, filename=filename, resume=opt.resume)
    else:
        loss_fn = YOLO_LOSS(model, save_logs=save_logs, rect_training=rect_training,
                            filename=filename, resume=opt.resume)

    evaluate = YOLO_EVAL(save_logs=save_logs, conf_threshold=config.CONF_THRESHOLD,
                         nms_iou_thresh=config.NMS_IOU_THRESH,  map_iou_thresh=config.MAP_IOU_THRESH,
                         device=config.DEVICE, filename=filename, resume=opt.resume)

    # starting epoch is used only when training is resumed by loading weights
    
    for epoch in range(starting_epoch, opt.epochs + starting_epoch):

        model.train()

        if not opt.only_eval:
            train_loop(model=model, loader=train_loader, loss_fn=loss_fn, optim=optim,
                       scaler=scaler, epoch=epoch, num_epochs=opt.epochs+starting_epoch,
                       multi_scale_training=not rect_training)

        model.eval()

        evaluate.check_class_accuracy(model, val_loader)

        evaluate.map_pr_rec(model, val_loader, anchors=model.head.anchors, epoch=epoch)

        # NMS WRONGLY MODIFIED TO TEST THIS FEATURE!!
        if not opt.nosaveimgs:
            save_predictions(model=model, loader=val_loader, epoch=epoch, num_images=5,
                             folder="SAVED_IMAGES", device=config.DEVICE, filename=filename,
                             labels=labels)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optim.state_dict(),
        }
        if not opt.nosavemodel:
            save_checkpoint(checkpoint, folder_path="./SAVED_CHECKPOINT", filename=filename, epoch=epoch)


if __name__ == "__main__":
    parser = arg_parser()
    main(parser)

