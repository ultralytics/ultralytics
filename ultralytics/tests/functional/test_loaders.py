from ultralytics.yolo.utils.checks import check_yaml
from ultralytics.yolo.utils.modeling.tasks import DetectionModel


def test_model_parser():
    cfg = check_yaml("../assets/dummy_model.yaml")  # check YAML

    # Create model
    model = DetectionModel(cfg)
    print(model)
    '''
    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()
    '''


if __name__ == "__main__":
    test_model_parser()
