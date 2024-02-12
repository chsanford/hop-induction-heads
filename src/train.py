from collections import defaultdict
from models import build_model
import numpy as np
import os
from tasks import get_task
import torch
import torch.nn.functional as F
from tqdm import tqdm
import uuid
import wandb
import yaml
import sys
from munch import Munch


def train(model: torch.nn.Module, task, rng, args):
    batch_size = args.training.batch_size
    num_iters = args.training.train_steps
    warmup = args.training.warmup_steps
    hooks = {}
    optim_fn = lambda model: torch.optim.AdamW(model.parameters(), lr=args.training.learning_rate)

    device = torch.device("mps")

    results = defaultdict(lambda: {})

    optimizer = optim_fn(model)

    def lr_func(t):
        if t <= warmup:
            return t/warmup
        else:
            return (num_iters-t)/(num_iters-warmup)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]

    pbar = tqdm(range(starting_step, args.training.train_steps))

    if 'samples' in args.training:
        all_inputs, all_targets = task.get_batch(args.training.samples)
        num_batches = args.training.samples // batch_size

    for t in pbar:
        if 'samples' in args.training:
            batch_idx = t % num_batches
            inputs = all_inputs[batch_idx*batch_size:(batch_idx+1)*batch_size]
            targets = all_targets[batch_idx*batch_size:(batch_idx+1)*batch_size]
        else:
            inputs, targets = task.get_batch(batch_size)
        inputs = inputs.to(device)
        targets = targets.to(device)
        preds = model(inputs)

        loss = F.cross_entropy(preds.reshape(-1, preds.shape[-1]), targets.reshape(-1))
        error = (preds.argmax(dim=-1) != targets).float().mean()


        results['train loss'][t] = loss.item()
        results['train error'][t] = error.item()

        with torch.no_grad():
            for key in hooks:
                results[key].append(hooks[key](model))

        model.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if t % args.wandb.log_every_steps == 0:
            eval_batch_size = 32
            eval_inputs, eval_targets = task.get_batch(eval_batch_size)
            eval_inputs = eval_inputs.to(device)
            eval_targets = eval_targets.to(device)
            eval_preds = model(eval_inputs)
            eval_loss = F.cross_entropy(eval_preds.reshape(-1, eval_preds.shape[-1]), eval_targets.reshape(-1))
            eval_error = (eval_preds.argmax(dim=-1) != eval_targets).float().mean()

            wandb_dict = {
                "loss": loss,
                "error": error,
                "eval loss": eval_loss,
                "eval error": eval_error,
            }

            for hops in range(task.min_hops, task.max_hops+1):
                hop_char = task.token_map.inv[hops - task.min_hops]
                relevant_seqs = (inputs[:, 0] == task.token_map[hop_char]).float().view(-1, 1)
                eval_relevant_seqs = (eval_inputs[:, 0] == task.token_map[hop_char]).float().view(-1, 1)
                if relevant_seqs.sum() != 0:
                    # compute error for relevant sequences
                    error_hops = ((preds.argmax(dim=-1) != targets).float() * relevant_seqs).sum() / relevant_seqs.sum() / task.seq_len
                    wandb_dict[f"error: {hops} hops"] = error_hops
                if eval_relevant_seqs.sum() != 0:
                    eval_error_hops = ((eval_preds.argmax(dim=-1) != eval_targets).float() * eval_relevant_seqs).sum() / eval_relevant_seqs.sum() / task.seq_len
                    wandb_dict[f"eval error: {hops} hops"] = eval_error_hops

            wandb.log(wandb_dict, step = t)

        pbar.set_description(f"loss {loss}, error {error}")

        if t % args.training.save_every_steps == 0:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": t,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and t % args.training.keep_every_steps == 0
            and t > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{t}.pt"))

    return results

def main(args):
    mps_device = torch.device("mps")

    wandb_name = f"{args.training.task}: depth {args.model.depth}, headcount {args.model.headcount}"
    if 'samples' in args.training:
        wandb_name += f", samples {args.training.samples}"
    wandb.init(
        dir=args.out_dir,
        project=args.wandb.project,
        config=args.__dict__,
        name=wandb_name,
        resume=True,
    )

    rng = np.random.RandomState(4444)
    task = get_task(
        args.training.task,
        rng=rng,
        **args.training.task_kwargs,
    )

    model = build_model(args.model, task.seq_len)
    model.to(mps_device)
    model.train()

    train(model, task, rng, args)


if __name__ == "__main__":
    # parse arguments, which contain --config, pointing to a config file
    # load in the contents of that file in a useful way
    # do not use quinine I'm mad at it

    # no quinine! just use argparse

    config_path = sys.argv[1]
    with open(config_path) as fp:
        args = Munch.fromDict(yaml.safe_load(fp))

    print(args)

    



    # parser = QuinineArgumentParser(schema=schema)
    # args = parser.parse_quinfig()
    # assert args.model.family in MODEL_LIST
    # print(f"Running with: {args}")

    run_id = str(uuid.uuid4())

    out_dir = os.path.join(args.out_dir, run_id)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    args.out_dir = out_dir

    with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
        yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)