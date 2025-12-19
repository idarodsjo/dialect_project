import torch
import torch.nn as nn
from transformers import WhisperForAudioClassification
from torchview import draw_graph

from transformers import WhisperProcessor

import numpy as np
from random import randint



def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = randint(0, len(wav) - sample_length - 1)
    return wav[random_offset : random_offset + sample_length]

# Example: simulate a 60s audio clip
wav = np.random.randn(60 * 16000)  # 60 seconds of fake audio



class WhisperWithRegression(nn.Module):
    """
    Wraps WhisperForAudioClassification (keeps encoder, mean-pool, 1024->256 projection, classifier),
    and adds a parallel regression head (256 -> 2) that predicts (lat_rad, lon_rad).
    """
    supports_gradient_checkpointing = True
    def __init__(self, base_ckpt: str, regression_weight: float = 1.0, regression_dropout: float = 0.1):
        super().__init__()
        self.base = WhisperForAudioClassification.from_pretrained(base_ckpt)
        self.regression_weight = regression_weight

        proj_dim = getattr(self.base.config, "classifier_proj_size", 256)
        self.regressor = nn.Sequential(
            nn.Dropout(regression_dropout),
            nn.Linear(proj_dim, 2)  # (lat_rad, lon_rad)
        )
        self._proj_cache = None  # set by forward hook
        self._hook = None
        self._register_proj_input_hook()

        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            """
            Forward the call to the wrapped HF model if it supports it.
            """
            base = getattr(self, "base_model", None)
            if base is not None and hasattr(base, "gradient_checkpointing_enable"):
                return base.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )
    
    def gradient_checkpointing_disable(self):
            base = getattr(self, "base_model", None)
            if base is not None and hasattr(base, "gradient_checkpointing_disable"):
                return base.gradient_checkpointing_disable()



    @property
    def config(self):
        # allow training script to access label2id/id2label, num_labels, etc.
        return self.base.config

    def freeze_feature_encoder(self):
        # keep parity with baseline flag
        if hasattr(self.base, "freeze_feature_encoder"):
            self.base.freeze_feature_encoder()

    def _register_proj_input_hook(self):
        """
        Register a forward hook on the classifier's final Linear layer so we can
        capture its *input* (the pooled+projected 256-d features) without re-implementing the head.
        """
        target_linear = None
        for m in self.base.modules():
            if isinstance(m, nn.Linear) and m.out_features == self.base.config.num_labels:
                target_linear = m
                break
        if target_linear is None:
            raise RuntimeError("Could not locate classifier Linear layer to hook.")
        def cache_input(module, inputs, output):
            # inputs is a tuple; take the tensor [B, 256]
            self._proj_cache = inputs[0]
        self._hook = target_linear.register_forward_hook(cache_input)

    def forward(self, input_features=None, attention_mask=None, labels=None, coords=None, **kwargs):
        # Run the base HF classification model to produce logits;
        # the forward hook will cache the 256-d features in self._proj_cache.
        base_out = self.base(
            input_features=input_features,
            attention_mask=attention_mask,
            labels=None,                 # we compute CE here for clarity
            output_hidden_states=False,
            return_dict=True,
        )
        logits = base_out.logits
        features_256 = self._proj_cache  # shape: (B, 256)

        reg_out = self.regressor(features_256) if features_256 is not None else None

        loss = None
        outputs = {"logits": logits, "regression": reg_out}

        if labels is not None:
            ce = self.ce(logits, labels)
            loss = ce
            outputs["ce"] = ce

        if coords is not None and reg_out is not None:
            mse = self.mse(reg_out, coords)
            loss = mse if loss is None else (loss + self.regression_weight * mse)
            outputs["mse"] = mse

        # ensure 'loss' exists for Trainer
        outputs["loss"] = loss if loss is not None else torch.tensor(0.0, device=logits.device)
        return outputs

if __name__ == "__main__":
    # simple test / visualize
    model = WhisperForAudioClassification.from_pretrained("NbAiLab/nb-whisper-medium")
    dummy_input = torch.randn(1, 80, 3000)  # Example shape for Whisper features
    graph = draw_graph(model, input_data=dummy_input, expand_nested=True)
    graph.visual_graph.render("nb_whisper_architecture", format="png")

