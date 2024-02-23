if __name__ == "__main__":
    import fiftyone as fo
    from ultralytics.utils.custom_utils.helpers import get_fiftyone_dataset

    dataset, classes = get_fiftyone_dataset(0)
    sesh = fo.launch_app(dataset)
    sesh.wait()