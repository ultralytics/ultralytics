






overrides=dict()




# for yoloe-26x-tp (provided by Jing with better performance than default hyperparameters)
or260105=dict(lr0=0.00038, lrf=0.88219, momentum=0.94751, weight_decay=0.00027, warmup_bias_lr=0.05684, warmup_epochs=0.98745, warmup_momentum=0.54064, box=9.83241, cls=0.64896, dfl=0.95824, hsv_h=0.01315, hsv_s=0.35348, hsv_v=0.19383, degrees=0.00012, translate=0.27484, scale=0.95, shear=0.00136, perspective=0.00074, flipud=0.00653, fliplr=0.30393, bgr=0.0, mosaic=0.99182, mixup=0.42713, cutmix=0.00082, copy_paste=0.40413, close_mosaic=10, o2m=0.70518, muon_w=0.4355, sgd_w=0.47908, cls_w=3.48357, epochs=15)
overrides["or260105"]=or260105

