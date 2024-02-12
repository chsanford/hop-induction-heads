from munch import Munch
import os
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
import yaml

device_name = 'mps'

def build_model(conf, seq_length):
    if conf.family == "gpt2_model":
        model = GPT2Encoder(
            vocab_size=conf.vocab_size,
            dim_embedding=conf.dim_embedding,
            seq_length=seq_length,
            depth=conf.depth,
            headcount=conf.headcount
        )
    return model

def get_model_from_run(run_path, step=-1, only_conf=False, seq_len=None):
    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path) as fp:
        conf = Munch.fromDict(yaml.safe_load(fp))
    if only_conf:
        return None, conf

    mps_device = torch.device(device_name)
    model = build_model(conf.model, seq_len).to(mps_device)

    if step == -1:
        state_path = os.path.join(run_path, "state.pt")
        state = torch.load(state_path, map_location=mps_device)
        model.load_state_dict(state["model_state_dict"])
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    return model, conf

class TransformerModel(nn.Module):
    def __init__(self,
                 vocab_size=16,
                 dim_embedding=16,
                 headcount=2,
                 depth=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim_embedding = dim_embedding
        self.headcount = headcount
        self.depth = depth

    def forward(self, xs):
        raise NotImplementedError


class GPT2Encoder(TransformerModel):
    def __init__(
        self,
        vocab_size=20,
        seq_length=10,
        dim_embedding=16, #128,
        depth=3, #12,
        headcount=2, #4
        ):
        super().__init__(
            vocab_size,
            dim_embedding,
            headcount,
            depth
        )
        self.seq_length = seq_length

        configuration = GPT2Config(
            vocab_size=vocab_size,
            n_positions=seq_length,
            n_embd=dim_embedding,
            n_layer=depth,
            n_head=headcount,
            use_cache=False,
            summary_type="cls_index"
        )
        self.backbone = GPT2Model(configuration)
        self.name = f"gpt2_embd={dim_embedding}_layer={depth}_head={headcount}"

    def forward(self, xs, output_attentions=False):
        output = self.backbone(xs, output_attentions=output_attentions)
        if output_attentions:
            return output.last_hidden_state, output.attentions
        else:
            return output.last_hidden_state
