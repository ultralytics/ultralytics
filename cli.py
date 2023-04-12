from FOD_YOLOv8.custom_trainer import CustomTrainer
from FOD_YOLOv8.custom_detector import CustomPredictor
import click

@click.group()
def cli():
    pass

@cli.command()
@click.option('--model', '-m', default='yolov5n.pt', help='Path to the model file')
@click.option('--data', type=str, required=False, default=None, help="Path to the data yaml file")
@click.option('--hyps', '-h', type=str, required=False, default=None, help="Path to the hyperparameter yaml file")
@click.option('--optimizer', '-o', type=str, default='SGD', help="Optimizer to be used for training. SGD by default")
@click.option('--imgsz', '-i', type=int, default='640', help="Size of the image. Default value is 640.")
@click.option('--batch', '-b', type=int, default=32, help="Batch size. Default value is 32.")
@click.option('--epochs', type=int, default=10, help="No. of epochs the model has to be trained. Default value is 10.")
@click.option('--evolve', '-e', type=bool, default=False, help="Use Optuna based TPE algorithm to evolve the hyperparametrs.")
@click.option('--device', '-d', type=str, default='cpu', help="CPU/Cuda Device to be used for training. Default value is 'cpu'.")
@click.option('--save', '-s', type=bool, default=True, help="Save the trained .pt file.")
@click.option('--trials', '-t', type=int, default=10, help="No. of trials to perform evolution on. Default value is 10")
@click.option('--study-name', type=str, default='Test', help="The study name for optuna hyperparameter tuning.")
@click.option('--resume', type=bool, default=False, help='Resume a pre-trained .pt file. Default set to False')
@click.option('--resume-model', type=str, default=None, help='The pt file that will be used to resume the training in case of evolve')
def train(model: str, data: str, hyps: str, optimizer: str, imgsz: int, batch: int, epochs: int, evolve: bool, device: str, save: bool, trials: int, study_name: str, resume: bool, resume_model: str):
    """
    model: Pass the model path.\n
    data: Pass the data file path.\n
    hyps: Pass the hyperparameter file path.\n
    optimizer: Specify the optimizer to be used. (`SGD` by default).\n
    imgsz: Pass the size of the image to be processed. (640 by default).\n
    batch: The batch size. (32 by default).
    epochs: Specify the no. of epochs. (10 by default).\n
    evolve: Do you want to use Optuna to perform hyperparameter/data augmentation tuning? Pass True or False. (False by default).\n
    device: CPU/Cuda Device to be used for training. (Default value is 'cpu').\n
    save: Save the trained .pt file. (Set to False by default). \n
    trials: No. of trials to perform evolution on. (10 by default).\n
    study name: The study name for optuna hyperparameter tuning. (Set to 'Test' by default). \n 
    """
    CustomTrainer(model_path=model, data_path=data, hyps_path=hyps, optimizer=optimizer, imgsz=imgsz, batch=batch, evolve=evolve, epochs=epochs, device=device, save=save, trials=trials, study_name=study_name, resume=resume, resume_model=resume_model)

@cli.command()
@click.option('--model', '-m', type=str, default='yolov5n.pt', help='Path to the model file')
@click.option('--source', '-s', type=str, required=True, help="Image/Video input source")
@click.option('--stream', type=bool, default=False)
@click.option('--rpi', type=bool, default=False)
@click.option('--device', '-d', type=str, default='cpu', help="CPU/Cuda Device to be used for training. Default value is 'cpu'.")
@click.option('--save', '-s', type=bool, default=False, help="Save the detection files.")
def detect(model: str, source: str, stream: bool, rpi: bool, device: str, save: bool):
    """
    model: Pass the model path.\n
    source: Pass the source of image or video input.\n
    stream: Is the input source a video stream? Pass True or False.\n
    rpi: Is the device this code is working on a Raspberry Pi? Pass True or False.\n
    CPU/Cuda Device to be used for training. Default value is 'cpu'.\n
    Save: Save the trained .pt file. Set to False by default. \n
    """
    CustomPredictor(model_path=model, source=source, is_stream=stream, is_rpi=rpi, device=device, save=save)

if __name__ == "__main__":
    cli()