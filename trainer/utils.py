class CheckpointStrategy:
    def __init__(self, config):
        self.config = config

    def save_checkpoint(self, model, optimizer, loss, step, is_master):
        raise NotImplementedError("This method should be implemented by subclasses")

    def load_checkpoint(self, model, optimizer, device, is_master):
        raise NotImplementedError("This method should be implemented by subclasses")