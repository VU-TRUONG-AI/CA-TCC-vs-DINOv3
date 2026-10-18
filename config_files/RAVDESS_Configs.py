from config_files.HAR_Configs import Config as HARConfig


class Config(HARConfig):
    def __init__(self):
        super().__init__()
        self.input_channels = 40
        self.num_classes = 8
        self.features_len = 77
        self.batch_size = 32
        self.num_epoch = 40
        self.audio_augmentation = True
