from m00nny_utils.util._trainer import _MetaTrainer
from m00nny_utils.parallel import load_transformers_as_sharded_module as tp_load


class LLMTrainer(_MetaTrainer):
    def __init__(self, args):
        super().__init__(args)

    def worker(self, _idx):
        super().worker(_idx)

        # Load the model and tokenizer
        self.model = tp_load(self.args.path, device_map="auto")
        self.tokenizer = self.model.tokenizer

        # Prepare the dataset and dataloader
        self.prepare_dataloader()

        # Prepare the optimizer and scheduler
        self.prepare_optimizer_and_scheduler()

        # Train the model
        self.train()
