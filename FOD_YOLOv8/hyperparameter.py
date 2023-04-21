import yaml
import math


class Hyperparameters:
    def __init__(self, path, default=None) -> None:
        self.path = path
        self.ranges = {
            'momentum': [0.9, 0.99],
            'weight_decay': [0.0001, 0.001],
            'warmup_momentum': [0.8, 0.95],
            'box': [0.1, 10],
            'cls': [0.1, 10],
            'dfl': [0.1, 10],
            'label_smoothing': [0, 1],
            'hsv_h': [-0.5, 0.5],
            'hsv_s': [-0.5, 0.5],
            'hsv_v': [-0.5, 0.5],
            'degrees': [-10, 10],
            'translate': [0, 0.5],
            'scale': [0, 1],
            'shear': [0, 1],
            'flipud': [0, 1],
            'fliplr': [0, 1],
            'mosaic': [0, 1],
            'mixup': [0, 1],
            'copy_paste': [0, 1]
        }
        if default is not None:
            self.saveHyps(default)
        else:
            self.update()
    
    def update(self):
        self.hyps = self.loadHyps()
        self.hyp_ranges_dict = self.create_hyp_ranges_dict()
        self.hyp_dict = self.create_hyp_dict()

    def loadHyps(self):
        with open(self.path, 'r') as file:
            return dict(yaml.load(file, Loader=yaml.FullLoader))
    
    def create_hyp_ranges_dict(self):
        # return {key: (self.hyps[key], math.floor(self.hyps[key]), math.ceil(self.hyps[key])) for key in self.hyps.keys()}
        hyps = {key: [True, self.hyps[key]] for key in self.hyps.keys()}
        for hyp in hyps:
            vals_list = hyps[hyp]
            if hyp in self.ranges:
                vals_list+=self.ranges[hyp]
            else:
                if math.floor(vals_list[1])==math.ceil(vals_list[1]):
                    vals_list[0] = False
                vals_list += [math.floor(vals_list[1]), math.ceil(vals_list[1])]
        return hyps
    
    def create_hyp_dict(self):
        return {key: self.hyps[key] for key in self.hyps.keys()}

    def get_hyps(self):
        return self.hyps
    
    def get_values_for_particular_hyp(self, key: str):
        return self.hyp_ranges_dict[key]
    
    def saveHyps(self, dict):
        with open(self.path, "w") as file:
            yaml.dump(dict, file)
        self.update()