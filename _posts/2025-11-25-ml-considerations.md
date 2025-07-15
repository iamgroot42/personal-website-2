---
layout: distill
title: Advice for working on ML projects
description: Lessons and recommendations based on my experiences working on ML projects (Python in general).
# tags: watermark removal, adversarial examples, diffusion models
date: 2025-11-25
thumbnail: assets/img/ml_consideration_squirrel.webp
published: false
citation: true
featured: true
categories: guide

authors:
  - name: Anshuman Suri
    url: "https://anshumansuri.com/"
    affiliations:
      name: Northeastern University

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Structuring your Codebase
  - subsections:
    - name: PIP-it!
    - name: Dataclasses are your friend
    - name: Generative Models
  - name: Evaluations
  - subsections:
    - name: How do you like them notifications?
    - name: Like a magic WAND(b)
  - name: I feel the need, the need for speed
  - subsections:
    - name: Compile "can" be your friend
    - name: Async transfers
    - name: Identify Bottlenecks
    - name: One batch, two batch, penny and dime
  - name: SLURM SLURM, Peralta
---

Working on ML projects in academia (and beyond) often feels like a constant battle between moving fast to test ideas and maintaining enough organization to actually make progress.

Based on my experiences across academia and industry, working with collaborators who have diverse coding backgrounds, and—perhaps most importantly—browsing through GitHub repositories of varying quality, I've picked up practices and design patterns that have genuinely transformed how I approach ML projects. These aren't abstract software engineering principles; they're tested techniques that have saved me from countless headaches and helped me move faster while making fewer mistakes.

In this blog, I'll document the lessons that have made the biggest difference in my day-to-day research workflow. Some might seem obvious in hindsight, others might challenge how you currently organize your work. Either way, I hope they help you spend less time wrestling with logistics and more time focused on the actual science. Note that this is a living document<d-footnote>This means I will update it every now and then based on new things I learn</d-footnote>—I'm constantly learning new tricks, and I'll add them here as I discover what works.

# Structuring your Codebase

Some experiments are straightforward and can be self-contained in a file or two. However, most ML projects that span a few weeks or more often end up with growing codebase sizes, with lots of reusable content that can bloat the overall project and lead to subtle inconsistencies when running experiments for different setups.

Let's say you've developed a new form of adversarial training and want to run experiments for varying perturbation strengths—including a baseline without any defense. Your project folder might look like this:

```bash
├── standard_training.py
├── adversarial_training.py
```

Now, during your standard training run, you notice the learning rate is too high (say you started with `1e-3`) and reduce it to `1e-4`, which fixes the issue. However, since you have separate files for adversarial and standard training, you forget to push the same update to the other file. Your experimental runs now differ not just in the presence/absence of adversarial training, but also in the optimizer hyperparameters—which can have non-trivial impacts on learning dynamics and final results.

This example might seem minor, but with growing project sizes and hyperparameters, it's easy to see how things can go wrong quickly. A straightforward solution would be to have a single `training.py` file and support standard training by setting the perturbation budget `epsilon` to 0 (or some other sentinel value). It could look something like:

```python
if args.epsilon == 0:
  # Standard training
  train(model, optim, data_loader)
else:
  # Adversarial training
  adv_train(model, optim, data_loader, args.epsilon)
```

This ensures the `model`, `optim`, and any other common components are used exactly the same way for both experiments. This approach is intuitive once you think about it, but I've seen many researchers<d-footnote>I've been guilty of this at several occasions.</d-footnote> and GitHub projects (especially academic ones) fall into the code duplication trap.

Going further with this example, I've also seen the equivalent of `adversarial_training_eps4.py` in the example above—creating duplicate files with nearly identical code and minor differences (mostly hyperparameters or datasets). This compounds the diverging changes problem and makes it hard to track what's actually different between experiments.

This "single point of failure" approach, in my opinion, is actually useful for research (as long as you catch the bugs, of course). For instance, let's say all your files use some common evaluation function:

```python
def evaluate(model, loader):
  acc = 0
  for x, y in loader:
    y_pred = model(x)
    acc += (y_pred == y).sum()
  model.train()
  return acc / len(loader)
```

There are two big issues here. First, the accuracy calculation is incorrect: `len(loader)` gives the number of batches, not the dataset size. Second, the model is returned to training mode after evaluation but is never set to eval mode before evaluation begins. This can be especially problematic when the model has data-dependent layers like batch normalization that accumulate statistics.

When the researcher catches this issue, they can at least be assured that whatever mistake they made invalidates all their experiments equally (requiring a complete redo), rather than having the same function in another file, correcting it only there, and making incorrect conclusions about which technique works better.

## PIP-it!

As your codebase grows and you start working on multiple related projects, you'll inevitably find yourself copy-pasting utility functions, model implementations, and evaluation scripts across different repositories. Let's say you've developed a novel membership inference attack for your latest paper. Six months later, you're working on a different project and want to use that same attack as a baseline or evaluation metric. What do you do? Copy the files over? Git submodule? Or worse, reimplement it from scratch because you can't find the exact version that worked?

This is where creating a proper Python package from your research code pays dividends. Not only does it make your life easier when reusing code across projects, but it also makes it significantly more likely that others will actually use your research. Think about it: would you rather download a ZIP file, dig through someone's experimental scripts, and try to figure out which functions are reusable, or would you prefer to simply `pip install` their package and import the functions you need? The latter is much more appealing, and higher adoption of your methods means more impact.

Here's how this evolution typically looks. You start with a project structure like this:

```bash
membership_inference_project/
├── train_target_model.py
├── run_mia_attack.py
├── utils.py  # Data loading, metrics, plotting
└── models.py # Target models and attack models
```

But then you realize that your attack implementation in `models.py` is generic enough that others could use it. Instead of letting this code rot in a single project folder, you can structure it as a proper package:

```bash
mia_toolkit/
├── setup.py
├── README.md
├── mia_toolkit/
│   ├── __init__.py
│   ├── attacks/
│   │   ├── __init__.py
│   │   └── membership_inference.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loaders.py
│   └── utils/
│       ├── __init__.py
│       └── metrics.py
```

With a minimal `setup.py`, you can now install this directly from GitHub. Note that I used the edit option to install the package with `-e` above: this is particularly useful for packages currently under development or when you want to make minimal changes to the code and don't want to reinstall the package every time you change something!

```python
# In your new project
pip install -e git+https://github.com/yourusername/mia-toolkit.git

# Clean imports in your code
from mia_toolkit.attacks import MembershipInferenceAttack
from mia_toolkit.data import load_private_dataset
```

The benefits extend beyond just your own convenience. When other researchers want to compare against your method, they don't need to reverse-engineer your experimental scripts—they can simply install your package and focus on the science. This dramatically lowers the barrier to adoption and increases the likelihood that your work will be built upon by others.

That said, don't go overboard with this. Not every 50-line script needs to become a package, and there's a delicate balance between making functions generic enough for reuse versus specific enough to actually be useful for your research. I typically package code when I find myself copy-pasting the same utilities across 2-3 projects, or when I think the methods are novel enough that others might want to use them as baselines.

A few practical notes: keep your package dependencies minimal and well-documented. I personally try to maintain one conda environment for most of my work, creating new ones only when external baselines require very specific package versions that would otherwise create conflicts. Also, resist the urge to over-engineer: your research package doesn't need to be production-ready software, it just needs to be clean and documented enough for other researchers to use.

## Dataclasses are your friend

I'll be honest—this is one of those "do as I say, not as I did" moments. If you look at some of my [older projects](https://github.com/suyeecav/model-targeted-poisoning/blob/342f35f7d1204c3a61e84b48c143ec819a55374c/dnn/mtp_dnn.py#L235), you'll see argument parsing that looks like this:

```python
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--model_arch', type=str, default='ResNet18')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--poison_lr', type=float, default=0.1)
parser.add_argument('--poison_momentum', type=float, default=0.9)
parser.add_argument('--poison_epochs', type=int, default=50)
parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--poison_fraction', type=float, default=0.1)
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--cuda_visible_devices', type=str, default='0')
parser.add_argument('--random_seed', type=int, default=0)
# ... and about 15 more arguments
```

This gets unwieldy fast, and worse, it's error-prone. What if you have both `args.lr` and `args.poison_lr`? It's easy to accidentally use the wrong one in your code, especially when you're debugging at 2 AM<d-footnote>Old habits: Bryan Johnson and Matthew Walker have convinced me to improve my sleeping habits. You should too- it makes a big difference!</d-footnote>.

Enter dataclasses with [SimpleParsing](https://github.com/lebrice/SimpleParsing)—a wrapper around argparse that leverages Python's dataclass functionality. Instead of the mess above, you can structure your arguments hierarchically:

```python
from dataclasses import dataclass
from simple_parsing import ArgumentParser

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    lr: float = 0.1                    # Learning rate for optimizer
    momentum: float = 0.9              # Momentum for SGD optimizer  
    weight_decay: float = 5e-4         # L2 regularization strength
    batch_size: int = 128              # Training batch size
    epochs: int = 200                  # Number of training epochs

@dataclass
class PoisonConfig:
    """Configuration for poisoning attack"""
    lr: float = 0.1                    # Learning rate for poison optimization
    momentum: float = 0.9              # Momentum for poison optimizer
    epochs: int = 50                   # Poison optimization epochs  
    fraction: float = 0.1              # Fraction of dataset to poison
    target_class: int = 0              # Target class for attack

@dataclass
class ExperimentConfig:
    """Overall experiment configuration"""
    data_dir: str = "data"             # Path to dataset directory
    model_arch: str = "ResNet18"       # Model architecture to use
    save_model: bool = False           # Whether to save trained model
    random_seed: int = 0               # Random seed for reproducibility

parser = ArgumentParser()
parser.add_arguments(TrainingConfig, dest="training")
parser.add_arguments(PoisonConfig, dest="poison") 
parser.add_arguments(ExperimentConfig, dest="experiment")

args = parser.parse_args()
```

Now you can run your script with clear, hierarchical arguments:

```bash
python train.py --training.lr 0.01 --poison.lr 0.1 --experiment.data_dir /path/to/data
```

The benefits are immediately obvious. No more confusion between `args.lr` and `args.poison_lr`—it's now `args.training.lr` versus `args.poison.lr`. The hierarchy makes it crystal clear which learning rate you're referring to, and the docstrings serve double duty as both code documentation and command-line help text.

But the real magic happens when you start reusing these configurations across files. Instead of copy-pasting argument definitions (and inevitably introducing inconsistencies), you can simply import your dataclasses:

```bash
# Your project structure
...
├── configs.py       # All dataclass definitions
├── train_model.py
├── evaluate_model.py
└── run_attack.py
```

Each script can import exactly the configurations it needs:

```bash
# In train_model.py
from configs import TrainingConfig, ExperimentConfig

# In run_attack.py  
from configs import PoisonConfig, ExperimentConfig

# In evaluate_model.py
from configs import ExperimentConfig
```

This ensures that when you update the default learning rate in `TrainingConfig`, it's automatically reflected across all scripts that use it. No more hunting through multiple files to make sure you've updated the same hyperparameter everywhere.

SimpleParsing also handles saving and loading configurations to/from YAML or JSON files, which makes experiment reproduction trivial. Instead of trying to remember the exact command-line arguments you used three weeks ago, you can simply:

```bash
# Save your current config to configs/experiment_1.yaml
# Reproduce the exact same experiment later
python train.py --config_path configs/experiment_1.yaml
```

# Evaluations

blah blah blah

## How do you like them notifications?

Picture this: you start a training run that's supposed to take 6 hours, close your laptop, and go about your day. Six hours later, you eagerly check back expecting to see beautiful loss curves, only to discover your script crashed 20 minutes in due to a CUDA out-of-memory error. Sound familiar?

Most ML experiments take hours or even days to complete, and the traditional approach of estimating runtime with `tqdm` and checking the ETA only gets you so far. What you really need is to know the moment your experiment finishes—or more importantly, when it crashes.

[knockknock](https://github.com/huggingface/knockknock) from HuggingFace has been an absolute lifesaver for this! It's a simple Python package that sends you notifications when your experiments complete or fail. The setup is straightforward:

```bash
pip install knockknock
```

You can use it as a decorator directly in your code but honestly, I prefer the command-line approach since it doesn't require modifying your existing code. You can set up a simple wrapper script in your `~/bin` directory:

```bash
#!/bin/bash
# Save this as ~/bin/knocky and make it executable with chmod +x
# Example below is for Telegram
knockknock telegram \
    --token YOUR_TELEGRAM_TOKEN \
    --chat-id YOUR_CHAT_ID \
    "$@"
```

Now you can run any experiment with notifications by simply prefixing your command:

```bash
# Instead of: python train_resnet.py --epochs 200
knocky python train_resnet.py --epochs 200
```

The beauty is that you get notifications both when your script completes successfully and when it crashes with an error. No more checking in every few hours or trying to estimate completion times. I personally use Telegram<d-footnote>Setup details for tokens and bot ID [here](https://github.com/huggingface/knockknock?tab=readme-ov-file#telegram)</d-footnote> since it's reliable and I always have it on my phone, but knockknock supports Slack, Discord, email, and several other platforms.

This simple change has saved me countless hours of babysitting experiments (or logging in anxiously every 1-2 hours). Plus, there's something deeply satisfying about getting a notification that your model finished training while you're grabbing coffee or on your way to work.

## Like a magic WAND(b)

Remember when comparing different experimental runs meant opening multiple terminal windows, squinting at loss values printed to stdout, and trying to remember which combination of hyperparameters gave you that promising result from last Tuesday? Or worse—frantically searching through your bash history because you forgot the exact arguments you used for your best-performing model?

I used to have training scripts that would dump metrics to text files, create matplotlib plots locally, and leave me manually tracking which experiment was which:

```python
def train_epoch(model, loader, optimizer, epoch, exp_name):
    total_loss = 0
    for batch_idx, (data, target) in enumerate(loader):
        # ... training code ...
        
        # Manual logging (the old way)
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            # Dump to files for later analysis
            with open(f'logs/{exp_name}_loss.txt', 'a') as f:
                f.write(f'{epoch},{batch_idx},{loss.item()}\n')
            
            # Save plots occasionally
            if batch_idx % 1000 == 0:
                plt.figure()
                plt.plot(losses)
                plt.savefig(f'plots/{exp_name}_loss_epoch_{epoch}.png')
                plt.close()
```

Then you end up with a mess of files like `resnet_lr001_wd0001_loss.txt` and `resnet_lr01_wd0005_loss.txt`, and good luck remembering which file corresponds to which exact experimental setup three weeks later.

Enter [Weights & Biases (wandb)](https://wandb.ai/)—hands down the biggest<d-footnote>TensorBoard and MLflow are good alternatives too; I just prefer wandb personally.</d-footnote> game-changer for my research workflow:

```python
import wandb

# Initialize once at the start of your script
wandb.init(
    project="my-awesome-research",
    config={
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "architecture": args.model,
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
    }
)

def train_epoch(model, loader, optimizer, epoch):
    for batch_idx, (data, target) in enumerate(loader):
        # ... training code ...
        
        # That's it! One line of logging
        wandb.log({
            "train/loss": loss.item(),
            "train/accuracy": accuracy,
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

# Automatically track your model's gradients and parameters
wandb.watch(model, log_freq=100)
```

<div class="row mt-1">
    <div class="col-sm mt-2 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/considerations/console.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

The magic isn't just in the simplicity of logging—it's in what wandb does with that information. Every single run gets tracked with:

- **All your hyperparameters**: No more "what learning rate did I use again?"
- **Real-time metrics**: Plots update live as your model trains
- **System monitoring**: GPU utilization, memory usage, CPU stats
- **Code tracking**: Git commit hash, diff, and even the exact command you ran
- **Reproducibility**: One-click to see the exact environment and arguments

But the real killer feature is **experiment comparison**. Instead of manually plotting loss curves from different text files, you can select multiple runs in the wandb interface and overlay their metrics instantly. Need to see how learning rate affects convergence? Select all runs with different LRs and compare their loss curves side-by-side. Want to find your best-performing model from the last month? Sort by validation accuracy and boom—there it is, with all the hyperparameters clearly listed.

You can even log media directly:

```python
# Log images, plots, and even 3D visualizations
wandb.log({
    "predictions": wandb.Image(prediction_plot),
    "confusion_matrix": wandb.plot.confusion_matrix(y_true, y_pred, labels),
    "sample_predictions": [wandb.Image(img, caption=f"Pred: {pred}") 
                          for img, pred in zip(sample_images, predictions)]
})
```

The filtering and search capabilities are phenomenal too. You can filter runs by any combination of hyperparameters, metric ranges, or tags. Looking for all ResNet experiments with learning rate > 0.01 that achieved >90% accuracy? Just use the built-in filters. This has saved me countless hours of digging through experimental logs<d-footnote>The free tier gives you unlimited personal projects and up to 100GB of storage, which is more than enough for most academic work</d-footnote>.

<div class="row mt-1">
    <div class="col-sm mt-2 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/considerations/filter.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Since adopting wandb, I've never lost track of an experimental run, never forgotten which hyperparameters produced good results, and never had to manually create comparison plots again. It's one of those tools that immediately makes you wonder how you ever lived without it.

# I feel the need, the need for speed

blah blah blah

## Compile "can" be your friend

blah blah blah

## Async transfers

You've probably noticed that your GPU utilization sometimes hovers around 70-80% instead of the near-100% you'd expect, even when your batch size and model complexity seem reasonable. The hidden culprit is often data transfer time between CPU and GPU—every `.to(device)` call is a synchronization point by default, meaning your expensive GPU sits idle waiting for data to crawl over the PCIe bus.

The easiest win is enabling pinned memory in your DataLoader, which uses page-locked host memory for much faster transfers:

```python
# Simple change with immediate benefits
train_loader = DataLoader(
    dataset, 
    batch_size=32, 
    pin_memory=True,  # This alone can give 20-30% speedup
    num_workers=4
)

# Now use non-blocking transfers
for data, target in train_loader:
    data = data.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)
    
    output = model(data)
    loss = criterion(output, target)
```

The real benefit comes when you can overlap transfers with computation:

```python
# X is large, Y is small
x = large_tensor.pin_memory()  # e.g., batch of images
y = small_tensor.pin_memory()  # e.g., single image or metadata

# Start transferring the large tensor asynchronously
x_gpu = x.cuda(non_blocking=True)

# While X is transferring, process Y
y_gpu = y.cuda()  # Small, transfers quickly
output_y = model2(y_gpu)

# By now X should be ready on GPU
output_x = model(x_gpu)
```

The key insight is using the time it takes to transfer large data to do other useful work—processing smaller tensors, running computations, or preparing the next batch<d-footnote>The PyTorch tutorial on pinned memory has more details on the underlying mechanics: https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html</d-footnote>.

**The crucial caveat**: async transfers only help when the next operation doesn't immediately depend on the transferred data. If you call `model(data)` right after `.to(device, non_blocking=True)`, PyTorch will still wait for the transfer to complete before starting the forward pass.

The real gotcha comes when transferring data back to CPU, especially with explicit async calls:

```python
def save_predictions(model, dataloader):
    predictions = []
    
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data.to(device))
            pred = output.argmax(dim=1)
            
            # If you use non_blocking=True here, this becomes dangerous:
            pred_cpu = pred.to('cpu', non_blocking=True)
            
            # BUG: numpy() might execute before transfer completes!
            predictions.extend(pred_cpu.numpy())  # Potential garbage data
```

The issue arises because when you explicitly use `non_blocking=True` for GPU→CPU transfers, the CPU doesn't wait for the transfer to complete. Accessing the data (like with `.numpy()`) before the transfer finishes gives you garbage. The fixes are straightforward:

```python
# Option 1: Don't use non_blocking for GPU→CPU (default behavior)
pred_cpu = pred.cpu()  # Synchronous by default
predictions.extend(pred_cpu.numpy())

# Option 2: If you do use non_blocking, explicitly synchronize
pred_cpu = pred.to('cpu', non_blocking=True)
torch.cuda.synchronize()  # Wait for all GPU operations to complete
predictions.extend(pred_cpu.numpy())

# Option 3: Accumulate on GPU, transfer once at the end
all_preds = torch.cat(gpu_predictions, dim=0).cpu().numpy()
```

The key insight is that async transfers shine when you can overlap them with computation that doesn't depend on the transferred data. Combined with pinned memory, this can substantially improve throughput for data-heavy workloads.

## Identify bottlenecks

blah blah blah

## One batch, two batch, penny and dime


# SLURM SLURM, Peralta

If you have access to a SLURM cluster, you're sitting on a goldmine for running ML experiments—but most people use it like an overpowered SSH session. Instead of thinking "how do I run this one experiment on SLURM?", start thinking "how do I run all my experiments efficiently?"

Here's what the inefficient approach looks like. You want to test your new membership inference attack:

```bash
sbatch experiment_1.slurm
# Wait... check results... then:
sbatch experiment_1.slurm
# Wait... check results... then:
sbatch experiment_1.slurm
# And so on...
```

There is no reason to submit jobs only when previous ones finish- in the absolute worst case (SLURM is extra busy, your jobs have very low priority in the queue), your jobs may actually end up running one after the other but in the average/best case, they will all run in parallel. tl;dr let the SLURM scheduler worry about scheduling the jobs- just submit them all at once!

One thing that is especially helpful here is job arrays—the feature that transforms SLURM from a glorified remote desktop into a proper experiment manager:

```bash
# One command to rule them all
sbatch --array=0-5 run_experiment.slurm
```

This single command launches 6 jobs simultaneously (indices 0 through 5), each with a unique `SLURM_ARRAY_TASK_ID` that your script can use to determine which specific experiment to run. Inside your `run_experiment.slurm`, you map the task ID to experimental parameters:

```bash
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=logs/exp_%A_%a.out
#SBATCH --error=logs/exp_%A_%a.err

# Define your experimental grid
MODELS=(resnet18 resnet50 vgg16)
DATASETS=(cifar10 imagenet)

# Calculate which model and dataset to use
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 2))
DATASET_IDX=$((SLURM_ARRAY_TASK_ID % 2))

MODEL=${MODELS[$MODEL_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}

echo "Running experiment: $MODEL on $DATASET"
python run_mia_attack.py --model $MODEL --dataset $DATASET
```

The `%A_%a` in the output files gives you the job array ID and task ID, so you get separate log files like `exp_12345_0.out`, `exp_12345_1.out`, etc. This makes debugging individual runs much easier than having everything mixed together.

But job arrays aren't just for hyperparameter sweeps. I use them for:

- **Testing different baselines**: Run your method against 10 different attack baselines simultaneously
- **Cross-dataset evaluation**: Evaluate the same model on multiple datasets
- **Ablation studies**: Test different components of your method (with/without data augmentation, different loss functions, etc.)
- **Robustness testing**: Same experiment with different random seeds to check variance

The real power comes when you need to run many variations. Want to test 5 models × 3 datasets × 4 random seeds = 60 experiments? Instead of submitting jobs one by one over several days, you submit one array job and walk away:

```bash
sbatch --array=0-59 comprehensive_eval.slurm
```

Your script maps the 60 task IDs to the appropriate combinations:

```bash
MODELS=(resnet18 resnet50 vgg16 densenet alexnet)
DATASETS=(cifar10 cifar100 imagenet)
SEEDS=(42 123 456 789)

# Extract indices from SLURM_ARRAY_TASK_ID
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 4))
DATASET_IDX=$(((SLURM_ARRAY_TASK_ID / 4) % 3))
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 12))

MODEL=${MODELS[$MODEL_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}
SEED=${SEEDS[$SEED_IDX]}
```

A few practical tips that have saved me headaches:

**Resource sizing**: Don't request more resources than you need. If your job only uses 8GB of memory, don't request 64GB—you'll wait longer in the queue and waste allocation budget. I usually run a few experiments locally first to get a rough estimate of memory and runtime requirements.

**Smart array sizing**: Instead of submitting massive arrays (like `--array=0-999`), consider breaking them into smaller chunks (`--array=0-99`, then `--array=100-199`, etc.). This gives you more flexibility if you need to cancel some jobs or if you discover an issue with your setup early on.

**Checkpoint your work**: For longer experiments, save intermediate results. SLURM has time limits, and there's nothing worse than losing 8 hours of training because your job hit the wall time. A simple checkpoint every epoch can save you from starting over.

As I mentioned in my [earlier post](https://www.anshumansuri.com/blog/2022/uva-rivanna/) about SLURM, there are plenty of other useful features and cluster-specific quirks to learn. But mastering job arrays alone will transform how you approach large-scale experimentation.

# Takeaways 

blah blah blah