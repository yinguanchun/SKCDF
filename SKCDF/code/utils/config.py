
class Config:
    def __init__(self,task):
        if task == "synapse":
            self.base_dir = '/data/ssd1/yinguanchun/Synapse'
            self.save_dir = '/data/ssd1/yinguanchun/Synapse'
            self.patch_size = (64, 128, 128)
            self.num_cls = 14
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 1500
        else: # amos
            self.base_dir = '/data/ssd1/yinguanchun/AMOS'
            self.save_dir = '/data/ssd1/yinguanchun/AMOS'
            self.patch_size = (64, 128, 128)
            self.num_cls = 16
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 1500


