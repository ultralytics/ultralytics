import yaml
import math


class Hyperparameters:
    def __init__(self, path, default=None) -> None:
        self.path = path
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
            if 'lr0' in hyp:
                vals_list+=3*vals_list[1:]
            elif 'momentum' in hyp:
                vals_list+=[0.9, 0.99]
            elif 'weight_decay' in hyp:
                vals_list+=[0.0001, 0.001]
            elif float(vals_list[1])==math.floor(vals_list[1]) and float(vals_list[1])==math.ceil(vals_list[1]):
                hyps[hyp]+=[vals_list[1]]*2
                hyps[hyp][0] = False
            else:
                vals_list += [math.floor(self.hyps[hyp]), math.ceil(self.hyps[hyp])]
    
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