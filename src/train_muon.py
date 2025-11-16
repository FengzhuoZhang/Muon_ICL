import os
import sys
with open(sys.argv[0]) as f:
    code = f.read()
from random import randint
import uuid
import argparse

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from eval import get_run_metrics
from tasks_muon import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema_muon import schema
from Muon import Muon
from models import build_model
import sys
sys.path.append("/home/aiops/zhangfz/Muon_linear_regression/Muon_ICL")
from debug_utils import setup_debugpy

import wandb

torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizers, loss_func):
    for optimizer in optimizers:
        optimizer.zero_grad()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    for optimizer in optimizers:
        optimizer.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds

def configure_muon(model, weight_decay, adam_lr, muon_lr, momentum=0.95, nesterov=False, ns_steps=5):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    
    # For Muon, we need to separate 2D parameters (which can be orthogonalized) 
    # from other parameters (which should use standard optimization)
    muon_params = []  # 2D parameters for Muon
    other_params = []  # other parameters for AdamW

    muon_name = []
    other_name = []
    for n, p in param_dict.items():
        # if "wte.weight" in n :
        #     other_params.append(p)
        #     other_name.append(n)
        #     continue

        if ("mlp" in n or "attn" in n) and p.dim() >= 2:  # 2D parameters (weight matrices)
            muon_params.append(p)
            muon_name.append(n)
        else:  # 1D parameters (biases, embeddings, etc.)
            other_params.append(p)
            other_name.append(n)

    # print("================================================\n")
    print(f"Muon parameters: {muon_name}\n")
    print(f"Other parameters: {other_name}\n")
    # print("================================================\n")
    
    # Create Muon optimizer for 2D parameters
    muon_optimizer = None
    if muon_params:
        muon_optimizer = Muon(
            params=muon_params,
            lr=muon_lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps
        )
    
    # Create AdamW optimizer for non-2D parameters
    adam_optimizer = None
    if other_params:
        # create optim groups for AdamW
        # decay_params = [p for p in other_params if p.dim() >= 2]
        # nodecay_params = [p for p in other_params if p.dim() < 2]
        optim_groups = [
            {'params': other_params, 'weight_decay': weight_decay},
            # {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        

        adam_optimizer = torch.optim.Adam(optim_groups, lr=adam_lr, betas=(0.9, 0.95))
    
    return [muon_optimizer, adam_optimizer]


def train(model, args, logfile):
    evaluation_step = 100
    eval_bsize = 1000
    if args.training.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.training.adam_lr, betas=(0.9, 0.95))
        optimizers = [optimizer]
    elif args.training.optimizer == "muon":
        optimizers = configure_muon(model, weight_decay=0.0, adam_lr=args.training.adam_lr, muon_lr=args.training.muon_lr, momentum=0.95, nesterov=False, ns_steps=5)
    else:
        raise ValueError(f"Invalid optimizer: {args.training.optimizer}")
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    # state_path = os.path.join(args.out_dir, "state.pt")

    n_dims = model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
        **args.tail,
    )
    eval_task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        eval_bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
        **args.tail,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        # if "sparse" in args.training.task:
        #     task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        # if num_training_examples is not None:
        #     assert num_training_examples >= bsize
        #     seeds = sample_seeds(num_training_examples, bsize)
        #     data_sampler_args["seeds"] = seeds
        #     task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs)

        loss_func = task.get_training_metric()

        loss, output = train_step(model, xs.cuda(), ys.cuda(), optimizers, loss_func)

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys.cuda()).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            if args.local_log:
                with open(logfile, "a") as f:
                    f.write(f"train loss at step {i}: {loss}" + "\n")
            else:
                wandb.log(
                    {
                        "overall_loss": loss,
                        "excess_loss": loss / baseline_loss,
                        "pointwise/loss": dict(
                            zip(point_wise_tags, point_wise_loss.cpu().numpy())
                        ),
                        "n_points": curriculum.n_points,
                        "n_dims": curriculum.n_dims_truncated,
                    },
                    step=i,
                )

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        # if i % args.training.save_every_steps == 0 and not args.test_run:
        #     training_state = {
        #         "model_state_dict": model.state_dict(),
        #         "optimizer_state_dict": optimizer.state_dict(),
        #         "train_step": i,
        #     }
        #     torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))

        if i % evaluation_step == 0:
            with torch.no_grad():
                xs = data_sampler.sample_xs(
                    curriculum.n_points,
                    eval_bsize,
                    curriculum.n_dims_truncated,
                    **data_sampler_args,
                )
                task = eval_task_sampler(**task_sampler_args)
                xs = xs.cuda()
                ys = ys.cuda()
                ys = task.eval_evaluate(eval_bsize, xs)
                loss_func = task.get_metric()
                output = model(xs, ys)
                loss = loss_func(output, ys)
                group_loss = task.process_loss(loss)
                print(f"group_loss at step {i}: {group_loss}")
                if args.local_log:
                    with open(logfile, "a") as f:
                        f.write(f"val loss at step {i}: {group_loss}" + "\n")
            


def main(args):
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 10000
    elif args.local_log:
        #TODO: implement local log
        logfile = os.path.join(args.out_dir, "log.txt")
        with open(logfile, "a") as f:
            f.write(code + "\n")
    else:
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )


    seed = args.training.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = build_model(args.model)
    # for name, param in model.named_parameters():
    #     print(name, param.shape)
    model.cuda()
    model.train()

    train(model, args, logfile)

    # if not args.test_run:
    #     _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    # setup_debugpy(force=True)
    # parser_add = argparse.ArgumentParser(description="NanoGPT Training Script with Muon")
    # parser_add.add_argument("--muon_lr", type=float, default=0.0)
    # parser_add.add_argument("--adam_lr", type=float, default=0.0)
    # parser_add.add_argument("--out_dir", type=str, default="")
    # args_add = parser_add.parse_args()

    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")

    # if args_add.muon_lr != 0.0:
    #     args.training.muon_lr = args_add.muon_lr
    # if args_add.adam_lr != 0.0:
    #     args.training.adam_lr = args_add.adam_lr
    # if args_add.out_dir != "":
    #     args.out_dir = args_add.out_dir

    if not args.test_run:
        optimizer = args.training.optimizer
        muon_lr = args.training.muon_lr
        adam_lr = args.training.adam_lr
        seed = args.training.seed
        if optimizer == "adam":
            exp_name = f"mode_adam_adam_lr_{adam_lr}_seed_{seed}"
        elif optimizer == "muon":
            exp_name = f"model_muon_muon_lr_{muon_lr}_seed_{seed}"
        else:
            raise ValueError(f"Invalid optimizer: {optimizer}")

        out_dir = os.path.join(args.out_dir, exp_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
