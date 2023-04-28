import yaml
import math
import json
import os

class Hyperparameters:
    def __init__(self, path, default=None) -> None:
        self.path = path
        if os.path.isfile(r'yolov8/utils/hyps.json'):
            f_path = r'yolov8/utils/hyps.json'
        else:
            f_path = r'utils/hyps.json'
        with open(f_path, 'r') as file:
            self.ranges = json.load(file)
        if default is not None:
            self.save_hyps(default)
        else:
            self.update()
    
    def update(self):
        self.hyps = self.load_hyps()
        self.hyp_ranges_dict = self.create_hyp_ranges_dict()
        self.hyp_dict = self.create_hyp_dict()

    def load_hyps(self):
        with open(self.path, 'r') as file:
            return dict(yaml.load(file, Loader=yaml.FullLoader))
    
    def create_hyp_ranges_dict(self):
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
    
    def save_hyps(self, dict):
        with open(self.path, "w") as file:
            yaml.dump(dict, file)
        self.update()
