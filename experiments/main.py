import torch
from PIL import Image
from matplotlib import pyplot as plt
from itertools import islice

from experiments.utils import show_image
from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.models.yolo.segment.train import SegmentationTrainer

st = SegmentationTrainer(cfg="./vdl-smart-trim-s-0-models-20230726-100018__yolo8m_imgsize_960/args.yaml", overrides=dict(close_mosaic=1))

best_weights = "./vdl-smart-trim-s-0-models-20230726-100018__yolo8m_imgsize_960/weights/best.pt"
arg = "./vdl-smart-trim-s-0-models-20230726-100018__yolo8m_imgsize_960/args.yaml"
data = "/Users/thomas/Documents/VBTI/Python/ultralytics/experiments/custom_dataset.yaml"

model = YOLO(best_weights, task='detect')


def get_image_from_dataloader(model, st, number, start):

    for i in range(number):
        # Somehow this dataloader still has augmentations, the model used below does not.
        dataloader = st.get_dataloader(
            "/Users/thomas/Documents/VBTI/DataAnalysis/autodl/dataset/yolo-converted/cucumber_dataset/train/images",
            batch_size=1)
        i += start
        # Get a new batch
        # normal_batch_iter = iter(dataloader)
        normal_batch = next(islice(dataloader, i, i + 1))
        normal_batch = st.preprocess_batch(normal_batch)

        normal_prediction = model.predict(normal_batch['img'])

        for r in normal_prediction:
            im_array = r.plot(labels=False, probs=False, masks=False,
                              boxes=False)  # plot a BGR numpy array of predictions
            img = Image.fromarray(im_array)  # RGB PIL image
            show_image(img, title=f"Natural Image-{i}",
                       path='/Users/thomas/Documents/School/TU:e/1. Master/Year 3/Graduation/Preparation Phase/Showcase/camera_test')
# get_image_from_dataloader(model, st, 10, 22)


def on_train_batch_start(trainer):
    with torch.no_grad():
        # Get dataloader from trainer
        dataloader = trainer.get_dataloader(
            "/Users/thomas/Documents/VBTI/DataAnalysis/autodl/dataset/yolo-converted/cucumber_dataset/train/images",
            batch_size=1)

        # Get a new batch
        normal_batch_iter = iter(dataloader)
        normal_batch = next(normal_batch_iter)
        normal_batch = trainer.preprocess_batch(normal_batch)

        normal_prediction = model.predict(normal_batch['img'])

        trainer.model.zero_grad()
        model.zero_grad()

        for r in normal_prediction:
            im_array = r.plot(labels=False, probs=False, masks=False,
                              boxes=False)  # plot a BGR numpy array of predictions
            img = Image.fromarray(im_array)  # RGB PIL image
            show_image(img, title="Natural Image",
                       path='/Users/thomas/Documents/School/TU:e/1. Master/Year 3/Graduation/Preparation Phase/Showcase/adv_3')


model.add_callback('on_train_batch_start', on_train_batch_start)
# Batch has to be 2, or different from callback above. Otherwise, complaining about reusing tensors
model.train(data=data, augment=False, rect=True, epochs=1, batch=1,
            hsv_h=0, hsv_s=0, hsv_v=0, degrees=0, translate=0, scale=0, shear=0, flipud=0, fliplr=0, mixup=0,
            agnostic_nms=False, cos_lr=False)


#  hsv_h=0, hsv_s=0, hsv_v=0, degrees=0, translate=0, scale=0, shear=0, flipud=0, fliplr=0, mixup=0, agnostic_nms=False, cos_lr=False, batch=1


# ------------------------------------------------------------- #
# # def on_train_batch_end(trainer):
# #     # Get dataloader from trainer
# #     dataloader = trainer.get_dataloader(
# #         "/Users/thomas/Documents/VBTI/DataAnalysis/autodl/dataset/yolo-converted/cucumber_dataset/train/images",
# #         batch_size=1)
# #
# #     # Get a new batch
# #     batch = next(iter(dataloader))
# #
# #     # def print_filename(filename):
# #     #     print(filename)
# #
# #     # Plot training samples
# #     plot_name = f"{trainer.epoch}-{trainer.tloss.shape[0]}"
# #     print(f"Plotting: {plot_name}")
# #     trainer.plot_training_samples(batch, plot_name)
# #     # batch_number = batch_number + 1
# #     # print(st.plots)
# #     # for i in range(5):
# #     #     # Get a batch of training data
# #     #     batch = next(iter(batch_sampler))
# #
# #
# #
# # def on_train_batch_end(trainer):
# #     # Retrieve the batch datai
# #     print()
# #     # print(dir(trainer))
# #     # ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__',
# #     #         '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__',
# #     #         '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__',
# #     #         '__subclasshook__', '__weakref__', '_do_train', '_setup_ddp', '_setup_train', 'accumulate', 'add_callback',
# #     #         'amp', 'args', 'batch_size', 'best', 'best_fitness', 'build_dataset', 'build_optimizer', 'build_targets',
# #     #         'callbacks', 'check_resume', 'csv', 'data', 'device', 'ema', 'epoch', 'epoch_time', 'epoch_time_start',
# #     #         'epochs', 'final_eval', 'fitness', 'get_dataloader', 'get_dataset', 'get_model', 'get_validator',
# #     #         'label_loss_items', 'last', 'lf', 'loss', 'loss_items', 'loss_names', 'metrics', 'model', 'on_plot',
# #     #         'optimizer', 'optimizer_step', 'plot_idx', 'plot_metrics', 'plot_training_labels', 'plot_training_samples',
# #     #         'plots', 'preprocess_batch', 'progress_string', 'resume', 'resume_training', 'run_callbacks', 'save_dir',
# #     #         'save_metrics', 'save_model', 'save_period', 'scaler', 'scheduler', 'set_callback', 'set_model_attributes',
# #     #         'setup_model', 'start_epoch', 'stop', 'stopper', 'test_loader', 'testset', 'tloss', 'train', 'train_loader',
# #     #         'train_time_start', 'trainset', 'validate', 'validator', 'wdir']
# #
# #     # print(trainer.loss)
# #     # print(trainer.data)
# #     # dl = trainer.get_dataloader()
# #     trainer.plot_training_samples(trainer.data, 1)
# #     trainer.plot_training_samples()
# #     # trainer.plot_metrics()
# #     # print(trainer.get_model("args.yaml", best_weights))
#
# # st.add_callback("on_train_batch_end", on_train_batch_end)
#
#
# # inputs = data['im_file'][0]
#
#
# # Forward pass the data through the model
# # Perform object detection on an image using the model
# # results = model(inputs)
# #
# # images = []
# # # Show the results
# # for r in results:
# #     im_array = r.plot()  # plot a BGR numpy array of predictions
# #     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
# #     im.show()  # show image
# #     # im.save('results.jpg')  # save image
# #     # images.append(im)
# #
# #
# # # FGSM attack code
# # def fgsm_attack(image, epsilon, data_grad):
# #     # Collect the element-wise sign of the data gradient
# #     sign_data_grad = data_grad.sign()
# #     # Create the perturbed image by adjusting each pixel of the input image
# #     perturbed_image = image + epsilon * sign_data_grad
# #     # Adding clipping to maintain [0,1] range
# #     perturbed_image = torch.clamp(perturbed_image, 0, 1)
# #     # Return the perturbed image
# #     return perturbed_image
