
from transformers import Trainer

class MultiTaskTrainer(Trainer):
    """
    Uses the model's 'loss' (combined CE + λ*MSE) returned from forward.
    Falls back gracefully if regression is off (no coords provided).
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss
