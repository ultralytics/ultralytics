from ultralytics import YOLO
import torch
import subprocess
import argparse

# Load a model
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)


# data_path = "/Users/sigurdbakkerud/Projects/specialization_project/data/data.yaml"
# model.to(torch.device("cpu"))
# results = model.train(data=data_path, epochs=1)  # train the model


def train(epochs, model, imgsz=640, device="0", batch_size=16, patience=30):

    

    data_path = '/home/vemund-sigurd/Documents/master/specialization_project/data/data.yaml'
    model_path = f'ultralytics/cfg/models/v8/{model}'
    # Define the command as a list of arguments
    command = [
        "yolo",
        "detect",
        "train",
        f"data={data_path}",
        f"model={model_path}",
        f"epochs={epochs}",
        f"imgsz={imgsz}",
        f"device={device}",
        "save_txt=True",
        f"batch={batch_size}",
        f"patience={patience}",
        "nms=True"
    ]

    # Run the command
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # model = YOLO('../ultralytics/cfg/models/v8/yolov8n-sigurd.yaml', "detect")  # build a new model from scratch

    parser = argparse.ArgumentParser(description="training parameters")

    # Add a list argument
    parser.add_argument(
        "--model",
        # nargs="+",  # '+' allows one or more values
        type=str,   # specify the type of elements in the list
        help="List of all models",
    )
    parser.add_argument(
        "--device",
        # nargs="+",  # '+' allows one or more values
        type=int,   # specify the type of elements in the list
        help="List of predictions to perform",
    )
    parser.add_argument(
        "--epochs",
        # nargs="+",  # '+' allows one or more values
        type=int,   # specify the type of elements in the list
        help="List of predictions to perform",
    )
    parser.add_argument(
        "--batch_size",
        # nargs="+",  # '+' allows one or more values
        type=int,   # specify the type of elements in the list
        help="List of predictions to perform",
    )


    args = parser.parse_args()
    model = args.model
    device = args.device
    epochs = args.epochs
    batch_size = args.batch_size
    train(epochs=epochs, model=model, device=device, batch_size=batch_size)