# CS336 Assignment 2 (systems): Systems and Parallelism

**Version 26.1.3**

CS336 Staff

Spring 2026

---

## 1 Assignment Overview

In this assignment, you will gain some hands-on experience with improving single-GPU training speed and scaling training to multiple GPUs.

### What you will implement

1. Benchmarking and profiling harness
2. Activation checkpointing
3. Flash Attention 2 Triton kernel
4. Distributed data parallel training
5. Optimizer state sharding
6. Fully sharded data parallel training

### What the code looks like

The assignment code and this write-up are available on GitHub at: <https://github.com/stanford-cs336/assignment2-systems>

Please clone the repository using Git. If there are any updates, we will notify you and you can `git pull` to get the latest.

1. `cs336-basics/`: In this assignment, you'll be profiling some of the components that we built in assignment 1. This folder contains the staff solution code for assignment 1, so you will find `cs336-basics/pyproject.toml` and the `cs336-basics/cs336_basics` package here. If you want to use your own implementation of the model, you can modify the `pyproject.toml` file in the base directory to point to your own package.
2. `/`: The `cs336-systems` base directory. We created an empty module named `cs336_systems`. Note that there's no code in here, so you should be able to do whatever you want from scratch.
3. `tests/*.py`: This directory contains all the tests you must pass. These tests invoke the hooks defined in `tests/adapters.py`. You'll implement the adapters to connect your code to the tests.
4. `README.md`: This file contains more details about the expected directory structure, as well as some basic instructions on setting up your environment.

### How to submit

You will submit the following files to Gradescope:

- **`writeup.pdf`**: Answer all the written questions. Please typeset your responses.
- **`code.zip`**: Contains all the code you've written.

Run the script in `test_and_make_submission.sh` to create the `code.zip` file.

---

## 2 Profiling and Benchmarking

In the first part of the assignment, we will explore how to optimize the performance of our Transformer model to make the most efficient use of the GPU. We will profile our model to understand where it spends time and memory during the forward and backward passes, then optimize the self-attention operation with custom GPU kernels, making it faster than is possible with regular PyTorch. In the subsequent parts of the assignment, we will leverage multiple GPUs and understand how to train a model across a cluster.

### 2.1 Profiling

Before implementing any optimizations, it is helpful to first profile our program to understand where it spends resources (e.g., time and memory). Otherwise, we risk optimizing parts of the model that don't account for significant time or memory, and therefore not seeing measurable end-to-end improvements.

We will implement three performance evaluation paths:

1. Simple end-to-end benchmarking using the Python standard library to time our forward and backward passes
2. Compute profiling with the NVIDIA Nsight Systems tool to understand how that time is distributed across operations on both the CPU and GPU
3. Memory profiling

#### 2.1.1 Setup - Importing your basics Transformer Model

Let's start by making sure that you can load the model from the previous assignment. In the previous assignment, we set up our model in a Python package, so that it could be easily imported later. We have added a reference implementation of the model in the `./cs336-basics` folder, and have pointed to it in the `pyproject.toml` file. By calling `uv run [command]` as usual, `uv` will automatically locate this local `cs336-basics` package. If you would like to use your own implementation of the model, you can modify the `pyproject.toml` file to point to your own package.

You can test that you can import your model with:

```bash
$ uv run python
Using CPython 3.13.13
Creating virtual environment at: /path/to/uv/env/dir
 Built cs336-systems @ file:///path/to/systems/dir
 Built cs336-basics @ file:///path/to/basics/dir
Installed 78 packages in 168ms
Python 3.13.13 (main, Apr 7 2026, 20:49:46) [Clang 22.1.1 ] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import cs336_basics
...
```

The relevant modules from assignment 1 should now be available (e.g., for `model.py`, you can import it with `import cs336_basics.model`).

#### 2.1.2 Model Sizing

Throughout this assignment, we will be benchmarking and profiling models to better understand their performance. To get a sense of how things change at scale, we will work with and refer to the following model configurations. For all models except in the leaderboard, we'll use a vocabulary size of 10,000 and a batch size of 4, with varying context lengths. This assignment (and later ones) will require a lot of results to be presented in tables and plots. We strongly recommend that you automate constructing tables for your writeup in code, since formatting tables in LaTeX or Typst can be very tedious. See `pandas.DataFrame.to_latex()` and `pandas.DataFrame.to_typst()` or write your own function to generate them from your preferred tabular representation.

| Size   | d_model | d_ff  | num_layers | num_heads |
|--------|---------|-------|------------|-----------|
| small  | 768     | 3072  | 12         | 12        |
| medium | 1024    | 4096  | 24         | 16        |
| large  | 1280    | 5120  | 36         | 20        |
| xl     | 2560    | 10240 | 32         | 32        |
| 10B    | 4608    | 12288 | 50         | 36        |

**Table 1:** Specifications of different model sizes. These are mostly based on GPT-2 configs. Use context length 512 unless otherwise specified.

#### 2.1.3 End-to-End Benchmarking

We will now implement a simple performance evaluation script. We will be testing many variations of our model (changing precision, swapping layers, etc.), so it will pay off to have your script enable these variations via command-line arguments to make them easy to run later on.

To start off, let's do the simplest possible profiling of our model by timing the forward pass, backward pass, and optimizer step. Since we will only be measuring speed and memory, it's fine to use random weights and data.

Measuring performance is subtle — some common traps can cause us to not measure what we want. For benchmarking GPU code, one caveat is that CUDA calls are asynchronous. When you call a CUDA kernel, such as when you invoke `torch.matmul`, the PyTorch function call returns control to your code without waiting for the matrix multiplication to finish. In this way, the CPU can continue running ahead and scheduling new operations while the GPU finishes the matrix multiplication, which is a major performance win. On the other hand, this means that naively measuring how long the `torch.matmul` call takes to return does not tell us how long the GPU takes to actually run the matrix multiplication. In PyTorch, we can call `torch.cuda.synchronize()` to wait for all scheduled GPU kernels to complete, allowing us to get more accurate measurements of CUDA kernel runtime. The synchronization in this operation refers to synchronizing the CPU runtime with the GPU runtime. With this in mind, let's write our basic profiling infrastructure.

**Problem (benchmarking_script): Benchmarking Script (4 points)**

(a) Write a script to perform basic end-to-end benchmarking of the forward pass, backward pass, and optimizer step in your model. Specifically, your script should support the following:

- Given hyperparameters (e.g., number of layers), initialize a model.
- Generate a random batch of data.
- Run $w$ warm-up steps (before you start measuring time), then time the execution of $n$ steps (either only forward, forward and backward, or forward and backward with optimizer step, depending on an argument). For timing, you can use the Python `timeit` module (e.g., either using the `timeit` function, or using `timeit.default_timer()`, which gives you the system's highest resolution clock, thus a better default for benchmarking than `time.time()`).
- Call `torch.cuda.synchronize()` after each step.

**Deliverable:** A script that will initialize a basics Transformer model with the given hyperparameters, create a random batch of data, and time forward-only, forward-and-backward, and full training steps that include the optimizer step.

(b) Time the forward, backward, and optimizer step for the model sizes described in Section 2.1.2. Use 5 warmup steps and compute the average and standard deviation of timings over 10 measurement steps. How long does a forward pass take? How about a backward pass? Do you see high variability across measurements, or is the standard deviation small?

**Deliverable:** A 1-2 sentence response with your timings.

(c) One caveat of benchmarking is not performing the warm-up steps. Repeat your analysis without the warm-up steps. How does this affect your results? Why do you think this happens? Also try to run the script with 1 or 2 warm-up steps. Why might the result still be different?

**Deliverable:** A 2-3 sentence response.

#### 2.1.4 Nsight Systems Profiler

End-to-end benchmarking does not tell us where our model spends time and memory during forward and backward passes, and so does not expose specific optimization opportunities. To know how much time our program spends in each component (e.g., function), we can use a profiler. An execution profiler instruments the code by inserting guards when functions begin and finish running, and thus can give detailed execution statistics at the function level (such as number of calls, how long they take on average, cumulative time spent on this function, etc).

Standard Python profilers (e.g., CProfile) are not able to profile CUDA kernels since these kernels are executed asynchronously on the GPU. Fortunately, NVIDIA ships a profiler that we can use via the CLI `nsys`. We recommend that you get an up-to-date version either from your package manager, or using the installers from their download page. In this part of the assignment, you will use `nsys` to analyze the runtime of your Transformer model.

Using `nsys` is straightforward: run your Python script from the previous section with `nsys profile` prepended. For example, you can run a basic profile for the script `benchmark.py` with:

```bash
$ uv run nsys profile -- python benchmark.py
```

You can then view the profile on your local machine with the NVIDIA Nsight Systems desktop application. Selecting a particular CUDA API call (on the CPU) in the **CUDA API** row of the profile will highlight all corresponding kernel executions (on the GPU) in the **CUDA HW** row.

A more comprehensive profiling run may look like:

```bash
$ uv run nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx \
    --pytorch=functions-trace,autograd-shapes-nvtx \
    --cudabacktrace=all \
    --python-backtrace=cuda \
    --gpu-metrics-devices=0 \
    -- python benchmark.py
```

**Figure 1:** A detailed Nsight Systems trace

In this example, `--trace` specifies which APIs to log, `--pytorch` inserts nvtx labels during module calls and autograd, `--cudabacktrace` and `--python-backtrace` give better backtraces to understand where in your code a given kernel was invoked from, and `--gpu-metrics-devices` specifies which GPU's utilization to measure.

Adding profiling to a run is not free, and will overall slow down your runs. It's often worth only enabling features you're looking for in a given run. Specifically, you might want to remove `--cudabacktrace=all` and `--python-backtrace=cuda` when tracebacks aren't needed, since they have outsize overhead.

We encourage you to experiment with various command-line options for `nsys profile` to get a sense of what it can do. You can also annotate your code with NVTX ranges, which will appear as blocks in the **NVTX** row of the profile capturing all CUDA API calls and associated kernel executions. In particular, you should use NVTX ranges to ignore the warm-up steps in your benchmarking script (by applying an `--nvtx-capture` filter on the nvtx label in the profile). You can also isolate which kernels are responsible for the forward and backward passes of your model, and you can even isolate which kernels are responsible for different parts of a self-attention layer by annotating your implementation as follows:

```python
import torch.cuda.nvtx as nvtx

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    ... # Q, K, V, mask
):
    ...
    with nvtx.range("computing attention scores"):
        ... # compute attention scores between Q and K
    with nvtx.range("computing softmax"):
        ... # compute softmax of attention scores
    with nvtx.range("final matmul"):
        ... # compute output projection
    return ...
```

You can swap your original implementation with the annotated version in your benchmarking script via:

```python
cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
```

Finally, it's worth noting that `torch.compile` can make it hard to attribute time and resources to specific parts of your code. You will likely have to wrap and strip various parts of your code in `torch.compile` and nvtx annotations to correctly attribute time and resource usage to various parts of your source.

**Problem (nsys_profile): Nsight Systems Profiling (5 points)**

Profile your forward pass, backward pass, and optimizer step using `nsys` with two model sizes from Table 1 of your choice as well as three power-of-two context lengths larger than 128, where the largest available size should be the longest context length you can fit in memory. Pick the combinations you think would be the most interesting to look at. For each profile answer the following questions:

(a) What is the total time spent on your forward pass? Does it match what we had measured before with the Python standard library?

**Deliverable:** A 1-2 sentence response.

(b) What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is this kernel invoked during a single forward pass of your model? Is it the same kernel that takes the most runtime when you do both forward and backward passes? (Hint: look at the "CUDA GPU Kernel Summary" under "Stats System View", and filter using NVTX ranges to identify which parts of the model are responsible for which kernels.)

**Deliverable:** A 1-2 sentence response.

(c) Although the vast majority of FLOPs take place in matrix multiplications, you will notice that several other kernels still take a non-trivial amount of the overall runtime. What other kernels besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the forward pass?

**Deliverable:** A 1-2 sentence response.

(d) Profile running one complete training step with your implementation of AdamW (i.e., the forward pass, computing the loss and running a backward pass, and finally an optimizer step, as you'd do during training). How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)? How about other kernels?

**Deliverable:** A 1-2 sentence response.

(e) Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer of your model during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?

**Deliverable:** A 1-2 sentence response.

#### 2.1.5 Mixed Precision

Up to this point in the assignment, we've been running with FP32 precision — all model parameters and activations have the `torch.float32` datatype. However, modern NVIDIA GPUs contain specialized GPU cores (Tensor Cores) for accelerating matrix multiplies at lower precisions. For example, the NVIDIA B200 spec sheet says that its maximum throughput with FP32 is 80 TFLOPS, while its maximum throughput with FP16 (half-precision floats) or BF16 (bfloat16) is significantly higher at a whopping 2500 TFLOPS. As a result, using lower-precision datatypes should help us speed up training and inference.

However, naively casting our model into a lower-precision format may come with reduced model accuracy. For example, many gradient values in practice are often too small to be representable in FP16, and thus become zero when naively training with FP16 precision. To combat this, it's common to use loss scaling when training with FP16 — the loss is simply multiplied by a scaling factor, increasing gradient magnitudes so they don't flush to zero. Furthermore, FP16 has a lower dynamic range than FP32, which can lead to overflows that manifest as a NaN loss. Full bfloat16 training is generally more stable (since BF16 has the same dynamic range as FP32), but can still affect final model performance compared to FP32.

To take advantage of the speedups from lower-precision datatypes, it's common to use mixed-precision training. In PyTorch, this is implemented with the `torch.autocast` context manager. In this case, certain operations (e.g., matrix multiplies) are performed in lower-precision datatypes, while other operations that require the full dynamic range of FP32 (e.g., accumulations and reductions) are kept as-is. For example, the following code will automatically identify which operations to perform in lower-precision during the forward pass and cast these operations to the specified data type:

```python
model : torch.nn.Module = ...  # e.g. your Transformer model
dtype : torch.dtype = ...      # e.g. torch.bfloat16
x : torch.Tensor = ...         # input data
with torch.autocast(device_type="cuda", dtype=dtype):
    y = model(x)
```

As alluded to above, it is generally a good idea to keep accumulations in higher precision even if the tensors themselves being accumulated have been downcast. The following exercise will help build your intuition as to why this is the case.

**Problem (mixed_precision_accumulation): Mixed-Precision Accumulation (1 point)**

Run the following code and comment on the accuracy of the results.

```python
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print(s)

s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print(s)
```

**Deliverable:** A 2-3 sentence response.

We will now apply mixed precision first to a toy model to build intuition and then to our benchmarking script.

**Problem (benchmarking_mixed_precision): Benchmarking Mixed Precision (2 points)**

(a) Consider the following model:

```python
class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x
```

Suppose we are training the model on a GPU and that the model parameters are originally in FP32. We'd like to use autocasting mixed precision with FP16. What are the data types of:

- the model parameters within the autocast context?
- the output of the first feed-forward layer (`ToyModel.fc1`)?
- the output of layer norm (`ToyModel.ln`)?
- the model's predicted logits?
- the loss?
- the model's gradients?

**Deliverable:** The data types for each of the components listed above.

(b) You should have seen that FP16 mixed precision autocasting treats the layer normalization layer differently than the feed-forward layers. What parts of layer normalization are sensitive to mixed precision? If we use BF16 instead of FP16, do we still need to treat layer normalization differently? Why or why not?

**Deliverable:** A 2-3 sentence response.

(c) Modify your benchmarking script to optionally run the model using mixed precision with BF16. Time the forward and backward passes with and without mixed-precision for each language model size described in Section 2.1.2. Compare the results of using full precision versus mixed precision, and comment on any trends as model size changes. You may find the `nullcontext` no-op context manager to be useful.

**Deliverable:** A 2-3 sentence response with your timings and commentary.

#### 2.1.6 Profiling Memory

So far, we have been looking at compute performance. We'll now shift our attention to memory, another major resource in language model training and inference. PyTorch also ships with a powerful memory profiler, which can keep track of allocations over time.

To use the memory profiler, you can modify your benchmarking script as follows:

```python
...  # warm-up phase in your benchmarking script
# Start recording memory history.
torch.cuda.memory._record_memory_history(max_entries=1000000)
...  # what you want to profile in your benchmarking script
# Save a pickle file to be loaded by PyTorch's online tool.
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
# Stop recording history.
torch.cuda.memory._record_memory_history(enabled=None)
```

This will output a file `memory_snapshot.pickle` that you can load into the following online tool: <https://pytorch.org/memory_viz>. This tool will let you see the overall memory usage timeline as well as each individual allocation that was made, with its size and a stack trace leading to the code where it originates.

**Problem (memory_profiling): Memory Profiling (4 points)**

Profile your complete training step of forward pass, backward pass, and optimizer step of the `xl` model from Table 1 with context lengths of 128 and 2048.

(a) Add an option to your profiling script to run your model through the memory profiler. It may be helpful to reuse some of your previous infrastructure (e.g., to activate mixed-precision, load specific model sizes, etc). Then, run your script to get a memory profile of the `xl` model when either doing inference only (just forward pass) or a full training step. What do your memory timelines look like? Can you tell which stage is running based on the peaks you see?

**Deliverable:** Two images of the "Active memory timeline" of an `xl` model, from the `memory_viz` tool: one for the forward pass, and one for running a full training step (forward and backward passes, then optimizer step), and a 2-3 sentence response.

(b) What is the peak memory usage of each context length when doing a forward pass? What about when doing a full training step?

**Deliverable:** A table with two numbers per context length.

(c) Find the peak memory usage of the `xl` model when using mixed-precision, for both a forward pass and a full training step. Does mixed-precision significantly affect memory usage?

**Deliverable:** A 2-3 sentence response.

(d) Consider the `xl` model. Given our reference hyperparameters, what is the size of a tensor of activations in the Transformer residual stream, in single-precision? Give this size in MiB (i.e., divide the number of bytes by $1024^2$).

**Deliverable:** A 1-2 sentence response with your derivation.

(e) Now look closely at the "Active Memory Timeline" from <https://pytorch.org/memory_viz> of a memory snapshot of the `xl` model doing a forward pass. When you reduce the "Detail" level, the tool hides the smallest allocations to the corresponding level (e.g., putting "Detail" at 10% only shows the 10% largest allocations). What is the size of the largest allocations shown? Looking through the stack trace, can you tell where those allocations come from?

**Deliverable:** A 1-2 sentence response.

(f) Nsight Systems also has flags for memory profiling. You can combine these with the Nsight flags from before to understand what allocations are happening at different steps in your model's lifespan. Use the PyTorch-provided NVTX labels to determine how much memory is saved for backward (these tensors are often called residuals) by a single `TransformerBlock` in your model. Note the 5 largest contributing operations, and what percentage of the overall memory they contribute. During the backward pass, all these tensors will be freed, but new gradient tensors are emitted at the same time. Based on your profiles showing how much memory was allocated during the forward pass, and how much memory usage changes for every `TransformerBlock` in the backward pass, calculate how much memory the produced gradient tensors for a `TransformerBlock` take. Does the result match what you expect?

**Deliverable:** Screenshots from Nsight Systems and a 1-2 paragraph response.

---

## 3 Single-GPU Memory

The later parts of this assignment will explore tricks to shard your tensors across multiple GPUs, but there are also tricks that can be applied even to single-GPU training. The most common of these is gradient checkpointing (also known as activation checkpointing).

### 3.1 Autograd Residuals

Recall that in order to perform a backward pass through your model, we need to save the activations that were produced in the forward pass. While this is obviously the case for some operations, by default it'll happen for many more than you might expect. The tensors saved for the backward pass are called "residuals", or simply "saved tensors".

Let's build some understanding of what's being saved in our network. Starting with our unassuming RMSNorm function (pure FP32 for simplicity), let's add some hooks for when tensors are being saved or retrieved by autograd.

```python
import torch
from torch import nn

x = torch.randn((4, 512, 2560), requires_grad=True)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5, device=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms
        return self.weight * x

def pack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

ln = RMSNorm(x.shape[-1])
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = ln(x)
    y.sum().backward()
```

The output shows a worrying amount of tensors being written out, several of them at full activation size!

```text
$ uv run scripts/autograd_experiment.py
Saving residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=<RsqrtBackward0 ...>
Saving residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=<RsqrtBackward0 ...>
Saving residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=<MulBackward0 ...>
Saving residual: shape=torch.Size([2560]), dtype=torch.float32, grad_fn=None
...
```

#### 3.1.1 Operator Fusion

In this case, it's clear that the granularity of the operations used is too high. We want a single op that takes in the RMSNorm weights and the activation, and spits out the output, as well as for that operation to be unitary in the backward pass. This is one motivation for kernel fusion. Since the RMSNorm is fairly well behaved, we can even automatically fuse it using `torch.compile`.

```python
...
ln = torch.compile(RMSNorm(x.shape[-1]))
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = ln(x)
    y.sum().backward()
```

The new output is significantly better:

```text
Saving residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([2560]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=None
```

We only need to save a single full-size activation tensor for the backward pass — namely, the input to the RMSNorm function.

### 3.2 Activation Checkpointing

While fusion is undoubtedly useful, it can only get us so far in saving memory. For instance, let's fuse a single `TransformerBlock` at size `xl`.

```python
import torch
from cs336_basics.model import RotaryEmbedding, TransformerBlock

d_model, d_ff, num_heads, context_length = 2560, 10240, 16, 2048
block = TransformerBlock(
    d_model=d_model, d_ff=d_ff, num_heads=num_heads,
    positional_encoder=RotaryEmbedding(dim=d_model // num_heads, context_length=context_length)
)
block = torch.compile(block, fullgraph=True)
x = torch.randn((4, context_length, d_model), requires_grad=True)
...
```

The script shows us how much memory we're saving for backward:

```text
Total size of saved tensors in single TransformerBlock: 3651.31 MiB
```

3.6 GiB for every layer. If we do this for all layers, we get 114 GiB of activations, just saved for backward!

#### 3.2.1 Recomputation

Instead of holding on to every tensor we generate, it's possible to save only periodic checkpoints of our results, and recompute the values in-between. `torch.utils.checkpoint.checkpoint` takes in a function, and arguments to that function. It then modifies the behavior of the function passed by:

1. In the forward pass:
   1. Saving the input values to the function
   2. Suppressing the saving of tensors in the forward pass
2. In the backward pass:
   1. Prepending a recomputation step where the forward pass is recomputed from the previously saved inputs, and values are saved for backward
   2. The backward pass is run and all tensors can be freed

```python
from torch.utils.checkpoint import checkpoint

def two_blocks(x):
    x = block(x)
    x = block(x)
    return x

def four_blocks_checkpoint(x):
    x = checkpoint(two_blocks, x, use_reentrant=False)
    x = checkpoint(two_blocks, x, use_reentrant=False)
    return x
```

**Problem (gradient_checkpointing): Memory-Optimal Gradient Checkpointing (4 points)**

Consider a Transformer with $N$ identical blocks stacked sequentially. Without any checkpointing, all $N$ blocks' worth of residuals are kept alive simultaneously, giving $O(N)$ peak activation memory.

(a) What checkpointing strategy minimizes peak activation memory, ignoring the compute cost? Describe how you would arrange the checkpoint calls (a code sketch is fine), and give the asymptotic peak activation memory and compute of your strategy as a function of $N$.

**Deliverable:** A 3-5 sentence description of the strategy and its asymptotic peak memory, plus a short code sketch.

(b) Consider the `xl` model config with batch size 4 and sequence length 2048 as above. If you only have the time/compute budget to run one step of recomputation (meaning you may not nest checkpoint calls), what is the best checkpointing strategy to reduce peak memory? Profile your run's peak memory to validate your hypothesis.

**Deliverable:** A 3-5 sentence description of your reasoning along with the measured peak memory for your strategy.


---

## 4 GPU Kernels

### 4.1 Optimizing Attention with FlashAttention-2

#### 4.1.1 Benchmarking PyTorch Attention

Your profiling likely suggests that there is an opportunity for optimization, both in terms of memory and compute, in your attention layers. At a high level, the attention operation consists of a matrix multiplication followed by softmax, then another matrix multiplication:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\text{mask}\left(\frac{QK^\top}{\sqrt{d_k}}\right)\right)V \tag{1}
$$

The naive attention implementation needs to save attention score matrices of shape $\text{seq\_len} \times \text{seq\_len}$ for each batch/head element, which can grow very large with long sequence lengths, causing out-of-memory errors for any tasks with long inputs or outputs. We will implement an attention kernel following the FlashAttention-2 paper, which computes attention by tiles and avoids ever explicitly materializing the $\text{seq\_len} \times \text{seq\_len}$ attention score matrices, enabling scaling to much longer sequence lengths.

**Problem (pytorch_attention): PyTorch Attention Benchmarking (2 points)**

(a) Benchmark your attention implementation at different scales. Write a script that will:

(i) Fix the batch size to 8 and don't use multihead attention (i.e. remove the head dimension).

(ii) Iterate through the cartesian product of [16, 32, 64, 128] for the head embedding dimension $d_{\text{model}}$, and [256, 1024, 4096, 8192, 16384] for the sequence length.

(iii) Create random inputs $Q, K, V$ for the appropriate size.

(iv) Time 100 forward passes through attention using the inputs.

(v) Measure how much memory is in use before the backward pass starts, and time 100 backward passes.

(vi) Make sure to warm up, and to call `torch.cuda.synchronize()` after each forward/backward pass.

**Deliverable:** A table with your timings, your calculations for the memory usage, and a 1-2 paragraph response.

### 4.2 Benchmarking JIT-Compiled Attention

Since version 2.0, PyTorch also ships with a powerful just-in-time compiler that automatically tries to apply a number of optimizations to PyTorch functions. In particular, it will try to automatically generate fused Triton kernels by dynamically analyzing your computation graph.

```python
layer = SomePyTorchModule(...)
compiled_layer = torch.compile(layer)
```

**Problem (torch_compile): Torch Compile (2 points)**

(a) Extend your attention benchmarking script to include a compiled version of your PyTorch implementation of attention, and compare its performance to the uncompiled version with the same configuration as the `pytorch_attention` problem above.

**Deliverable:** A table comparing your forward and backward pass timings for your compiled attention module with the uncompiled version.

(b) Now, compile your entire Transformer model in your end-to-end benchmarking script. How does the performance change?

**Deliverable:** A table comparing your vanilla and compiled Transformer model.

#### 4.2.1 Example - Weighted Sum

To introduce what you'll need to know about Triton and how it interoperates with PyTorch, we will work through an example kernel for a "weighted sum" operation.

Given an input matrix $X$, we'll multiply its entries by a column-wise weight vector $w$, and sum each row, giving us the matrix-vector product of $X$ and $w$.

```python
def weighted_sum(x, weight):
    # Here, assume that x has n-dim shape [..., D], and weight has 1D shape [D]
    return (weight * x).sum(axis=-1)
```

When writing our Triton kernel, we'll have each program instance (potentially running in parallel) compute the weighted sum of a tile of rows of $x$, and write the corresponding scalar outputs to the output tensor. We'll use the block pointer abstraction with `tl.make_block_ptr` to greatly simplify the pointer arithmetic.

```python
import triton
import triton.language as tl

@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr,
    output_ptr,
    x_stride_row, x_stride_dim,
    weight_stride_dim,
    output_stride_row,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D,),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(NUM_ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        output += tl.sum(row * weight[None, :], axis=1)
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))

    tl.store(output_block_ptr, output, boundary_check=(0,))
```

Let's now wrap this kernel in a PyTorch Autograd function:

```python
class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        D, output_dims = x.shape[-1], x.shape[:-1]
        input_shape = x.shape
        x = rearrange(x, "... d -> (...) d")
        ctx.save_for_backward(x, weight)
        ...
        y = torch.empty(output_dims, device=x.device)
        n_rows = y.numel()
        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight, y,
            x.stride(0), x.stride(1),
            weight.stride(0), y.stride(0),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE,
        )
        return y.view(input_shape[:-1])

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        ...
        return grad_x, grad_weight
```

For the backward pass, we need to compute gradients:

$$
(\nabla_x\mathcal{L})_{ij} = w_j \cdot (\nabla_{f(x,w)}\mathcal{L})_i \tag{2}
$$

$$
(\nabla_w\mathcal{L})_j = \sum_{i=1}^{n} x_{ij} \cdot (\nabla_{f(x,w)}\mathcal{L})_i \tag{3}
$$

#### 4.2.2 FlashAttention-2 Forward Pass

You will replace your PyTorch attention implementation with a significantly improved Triton implementation following FlashAttention-2 [T. Dao, 2023].

Recall the forward pass for attention:

$$
S = QK^\top / \sqrt{d} \tag{4}
$$
$$
P_{ij} = \text{softmax}_j(S)_{ij} \tag{5}
$$
$$
O = P V \tag{6}
$$

The standard backward pass is:

$$
dV = P^\top dO \tag{7}
$$
$$
dP = dO V^\top \tag{8}
$$
$$
dS_i = \text{dsoftmax}(dP_i) = (\text{diag}(P_i) - P_i P_i^\top) dP_i \tag{9}
$$
$$
dQ = dS K / \sqrt{d} \tag{10}
$$
$$
dK = dS^\top Q / \sqrt{d} \tag{11}
$$

The main goal of FlashAttention is to avoid reading and writing the attention matrix to and from HBM, to reduce IO and peak memory costs. This is accomplished using three techniques: **tiling**, **recomputation**, and **operator fusion**.

The logsumexp is defined as:

$$
L_i = \log\left(\sum_j \exp(S_{ij})\right) \tag{12}
$$

**Backward pass with recomputation:**

Using $L$ and $D = \text{rowsum}(O \circ dO)$, the backward pass can be computed without softmax:

$$
S = QK^\top / \sqrt{d} \tag{13}
$$
$$
P_{ij} = \exp(S_{ij} - L_i) \tag{14}
$$
$$
dV = P^\top dO \tag{15}
$$
$$
dP = dO V^\top \tag{16}
$$
$$
dS_{ij} = P_{ij} (dP_{ij} - D_i) \tag{17}
$$
$$
dQ = dS K / \sqrt{d} \tag{18}
$$
$$
dK = dS^\top Q / \sqrt{d} \tag{19}
$$

**Algorithm 1:** FlashAttention-2 forward pass

**Require:** $Q \in \mathbb{R}^{N_q \times d}$, $K, V \in \mathbb{R}^{N_k \times d}$, tile sizes $B_q, B_k$

1. Split $Q$ into $T_q = \lceil N_q / B_q \rceil$ tiles $Q_1, \dots, Q_{T_q}$ of size $B_q \times d$
2. Split $K, V$ into $T_k = \lceil N_k / B_k \rceil$ tiles $K^{(1)}, \dots, K^{(T_k)}$ and $V^{(1)}, \dots, V^{(T_k)}$ of size $B_k \times d$
3. **for** $i = 1, \dots, T_q$ **do**
4. Load $Q_i$ from global memory
5. Initialize $O_i^{(0)} = \mathbf{0} \in \mathbb{R}^{B_q \times d}$, $l_i^{(0)} = 0 \in \mathbb{R}^{B_q}$, $m_i^{(0)} = -\infty \in \mathbb{R}^{B_q}$
6. **for** $j = 1, \dots, T_k$ **do**
7. Load $K^{(j)}, V^{(j)}$ from global memory
8. Compute $S_i^{(j)} = Q_i (K^{(j)} / \sqrt{d})^\top \in \mathbb{R}^{B_q \times B_k}$
9. Compute $m_i^{(j)} = \max(m_i^{(j-1)}, \text{rowmax}(S_i^{(j)})) \in \mathbb{R}^{B_q}$
10. Compute $\tilde{P}_i^{(j)} = \exp(S_i^{(j)} - m_i^{(j)}) \in \mathbb{R}^{B_q \times B_k}$
11. Compute $l_i^{(j)} = \exp(m_i^{(j-1)} - m_i^{(j)}) l_i^{(j-1)} + \text{rowsum}(\tilde{P}_i^{(j)}) \in \mathbb{R}^{B_q}$
12. Compute $O_i^{(j)} = \text{diag}(\exp(m_i^{(j-1)} - m_i^{(j)})) O_i^{(j-1)} + \tilde{P}_i^{(j)} V^{(j)}$
13. **end for**
14. Compute $O_i = \text{diag}(l_i^{(T_k)})^{-1} O_i^{(T_k)}$
15. Compute $L_i = m_i^{(T_k)} + \log(l_i^{(T_k)})$
16. Write $O_i$ to global memory as the $i$-th tile of $O$.
17. Write $L_i$ to global memory as the $i$-th tile of $L$.
18. **end for**
19. **Return** the output $O$ and the logsumexp $L$.

**Triton Tips and Tricks**

- You can use print statements in Triton with `tl.device_print` to debug: <https://triton-lang.org/main/python-api/generated/triton.language.device_print.html>.
- When defining block pointers, make sure they have the correct offsets.
- The launch grid of thread blocks is set with `kernel_fn[(launch_grid_d1, launch_grid_d2, ...)](...arguments...)`.
- Perform matrix multiplications with `tl.dot`.
- To advance a block pointer, use `*_block_ptr = *_block_ptr.advance(...)`.

**Problem (flash_forward): FlashAttention-2 Forward Pass (15 points)**

(a) Write a pure PyTorch (no Triton) `autograd.Function` that implements the FlashAttention-2 forward pass. Your implementation should take input $Q$, $K$, and $V$ as well as a flag `is_causal` and produce the output $O$ and the logsumexp value $L$.

**Deliverable:** A `torch.autograd.Function` subclass that implements FlashAttention-2 in the forward pass. Implement `adapters.get_flashattention_autograd_function_pytorch`, then run `uv run pytest -k test_flash_forward_pass_pytorch`.

(b) Write a Triton kernel for the forward pass of FlashAttention-2 following Algorithm 1. Then, write another subclass of `torch.autograd.Function` that calls this (fused) kernel in the forward pass.

```python
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    ...
```

**Deliverable:** A `torch.autograd.Function` subclass that implements FlashAttention-2 in the forward pass using your Triton kernel. Implement `adapters.get_flash_autograd_function_triton`, then run `uv run pytest -k test_flash_forward_pass_triton`.

(c) Add a flag for causal masking. When set to `True`, enable index comparison for causal masking.

**Deliverable:** An additional flag for your `torch.autograd.Function` subclass that implements causal masking. Default to `False` so previous tests still pass.

**Problem (flash_backward): FlashAttention-2 Backward Pass (5 points)**

Implement the backward pass for your FlashAttention-2 `autograd.Function` using PyTorch (not Triton) and `torch.compile`. Your implementation should take $Q$, $K$, $V$, $O$, $dO$, and $L$ tensors as inputs, and return $dQ$, $dK$ and $dV$.

**Deliverable:** Run `uv run pytest -k test_flash_backward`.

**Problem (flash_benchmarking): FlashAttention-2 Benchmarking (5 points)**

(a) Write a benchmarking script using `triton.testing.do_bench` that compares the performance of your Triton implementation of FlashAttention-2 with a regular PyTorch implementation.

**Deliverable:** A table of results comparing your implementation of FlashAttention-2 with the PyTorch implementation, reporting forward, backward, and end-to-end latencies.

#### 4.2.3 OPTIONAL: Triton backward pass

Algorithm 2 shows the FlashAttention-2 backward pass as it should be implemented in Triton.

**Algorithm 2:** Tiled FlashAttention-2 backward pass

**Require:** $Q, O, dO \in \mathbb{R}^{N_q \times d}$, $K, V \in \mathbb{R}^{N_k \times d}$, $L \in \mathbb{R}^{N_q}$, tile sizes $B_q, B_k$

1. Compute $D = \text{rowsum}(O \circ dO) \in \mathbb{R}^{N_q}$
2. Split $Q, O, dO$ into $T_q = \lceil N_q / B_q \rceil$ tiles of size $B_q \times d$
3. Split $K, V$ into $T_k = \lceil N_k / B_k \rceil$ tiles of size $B_k \times d$
4. Split $L, D$ into $T_q$ tiles of size $B_q$
5. **for** $j = 1, \dots, T_k$ **do**
6. Load $K^{(j)}, V^{(j)}$ from global memory
7. **for** $i = 1, \dots, T_q$ **do**
8. Compute $S_i^{(j)} = Q_i (K^{(j)} / \sqrt{d})^\top$
9. Compute $P_i^{(j)} = \exp(S_i^{(j)} - L_i)$
10. Compute $dV_{(i)}^{(j)} = dV_{(i-1)}^{(j)} + (P_i^{(j)})^\top dO_i$
11. Compute $dP_{(i)}^{(j)} = dO_i (V^{(j)})^\top$
12. Compute $dS_{(i)}^{(j)} = P_i^{(j)} \circ (dP_{(i)}^{(j)} - D_i)$
13. Compute $dK_{(i)}^{(j)} = dK_{(i-1)}^{(j)} + (dS_{(i)}^{(j)})^\top Q_i / \sqrt{d}$
14. **end for**
15. **end for**
16. **for** $i = 1, \dots, T_q$ **do**
17. **for** $j = 1, \dots, T_k$ **do**
18. Compute $S_i^{(j)} = Q_i (K^{(j)} / \sqrt{d})^\top$
19. Compute $P_i^{(j)} = \exp(S_i^{(j)} - L_i)$
20. Compute $dQ_{(i)}^{(j)} = dQ_{(i)}^{(j-1)} + dS_{(i)}^{(j)} K^{(j)} / \sqrt{d}$
21. **end for**
22. **end for**
23. **Return** $dQ, dK, dV$.


---

## 5 Distributed Data Parallel Training

In this next part of the assignment, we'll explore methods for using multiple GPUs to train our language models, focusing on data parallelism. We'll start with a primer on distributed communication in PyTorch. Then, we'll study a naive implementation of distributed data parallel training, then implement and benchmark various improvements to communication efficiency.

### 5.1 Single-Node Distributed Communication in PyTorch

Let's start by looking at a simple distributed application in PyTorch.

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size):
    setup(rank, world_size)
    data = torch.randint(0, 10, (3,))
    print(f"rank {rank} data (before all-reduce): {data}")
    dist.all_reduce(data, async_op=False)
    print(f"rank {rank} data (after all-reduce): {data}")

if __name__ == "__main__":
    world_size = 4
    mp.spawn(fn=distributed_demo, args=(world_size, ), nprocs=world_size, join=True)
```

**Terminology**

- **node**: a machine on the network.
- **worker**: an instance of a program participating in distributed training.
- **world size**: The number of total workers in a process group.
- **global rank**: An integer ID (between 0 and `world_size-1`) that uniquely identifies a worker.
- **local world size**: The number of workers running locally on a given node.
- **local rank**: An integer ID (between 0 and `local_world_size-1`) that uniquely identifies a local worker.

#### 5.1.1 Best Practices for Benchmarking Distributed Applications

- Whenever possible, run benchmarks on the same machine to facilitate controlled comparisons.
- Perform several warm-up steps before timing. 5 iterations of warmup is generally sufficient.
- Call `torch.cuda.synchronize()` to wait for CUDA operations to complete when benchmarking on GPUs.
- Timings may vary across ranks; use `dist.all_gather_object` to collect results from all ranks.
- Debug locally with Gloo on CPU, benchmark with NCCL on GPU.

**Problem (distributed_communication_single_node): Distributed Communication (Single Node) (5 points)**

Write a script to benchmark the runtime of the all-reduce operation in the single-node multi-process setup. Experiment with:
- **all-reduce data size**: float32 tensors of 1MB, 10MB, 100MB, 1GB.
- **Number of GPUs/processes**: 2, 4, or 6.

**Deliverable:** Plot(s) and/or table(s) comparing the various settings, with 2-3 sentences of commentary.

### 5.2 A Naive Implementation of Distributed Data Parallel Training

Data parallelism splits batches across multiple devices (e.g., GPUs), enabling training on large batch sizes that do not fit on a single device.

Steps for naive distributed data parallel training:

1. Each device constructs a (randomly-initialized) model. Broadcast initial parameters from rank 0.
2. Given a batch with $n$ examples, each device receives $n/d$ disjoint examples.
3. Each device runs forward and backward passes on its $n/d$ examples.
4. All-reduce gradients across devices to average them.
5. Each device runs an optimizer step to update its copy of the parameters.

**Problem (naive_ddp): Naive DDP (5 points)**

**Deliverable:** Implement a naive form of distributed data parallel training that all-reduces individual parameter gradients after the backward pass. Implement `adapters.get_ddp`, then run `uv run pytest tests/test_ddp.py`.

**Problem (naive_ddp_benchmarking): Naive DDP Benchmarking (3 points)**

Benchmark your language model when trained with this naive DDP. Measure the total time per training step and the proportion of time spent on communicating gradients. Use 1 node x 2 GPUs, `xl` model size.

**Deliverable:** A description of your benchmarking setup, along with the measured time per training iteration and time spent communicating gradients.

### 5.3 Improving Upon the Minimal DDP Implementation

The minimal DDP implementation has two key limitations:

1. It conducts a separate all-reduce for every parameter tensor.
2. It waits for the backward pass to finish before communicating gradients.

#### 5.3.1 Reducing the Number of Communication Calls

Rather than issuing a communication call for each parameter tensor, batch gradients into a single tensor before the all-reduce. Use `torch._utils._flatten_dense_tensors` and `torch._utils._unflatten_dense_tensors`.

**Problem (minimal_ddp_flat_benchmarking): Minimal DDP with Flat Gradients Benchmarking (2 points)**

**Deliverable:** The measured time per training iteration and time spent communicating gradients with a single batched all-reduce call. 1-2 sentences comparing results.

#### 5.3.2 Overlapping Computation with Communication of Individual Parameter Gradients

Use backward hooks (`register_post_accumulate_grad_hook`) and asynchronous communication (`async_op=True`) to all-reduce parameter gradients as soon as they're ready during the backward pass.

**Problem (ddp_overlap_individual_parameters): DDP with Overlapping Individual Parameters (5 points)**

Implement a Python class for distributed data parallel training that overlaps gradient communication and backward pass computation. Public interface:

- `def __init__(self, module: torch.nn.Module)`
- `def forward(self, *inputs, **kwargs)`
- `def finish_gradient_synchronization(self)`

**Deliverable:** Implement the container class. Implement adapters `adapters.get_ddp` and optionally `adapters.ddp_on_after_backward`. Run `uv run pytest tests/test_ddp.py`.

**Problem (ddp_overlap_individual_parameters_benchmarking): DDP Overlapping Individual Parameters Benchmarking (1 point)**

(a) Benchmark the performance of your DDP implementation when overlapping backward pass computation with communication.

**Deliverable:** The measured time per training iteration, with 1-2 sentences comparing the results.

(b) Instrument your benchmarking code with the Nsight profiler, comparing the initial DDP implementation with this overlapped implementation.

**Deliverable:** 2 screenshots demonstrating that communication is or isn't overlapped with the backward pass.


---

## 6 Optimizer State Sharding

Distributed data parallel training requires each rank to hold a distinct copy of the model parameters and optimizer state. The AdamW optimizer maintains two floats per parameter, consuming twice as much memory as the model weights. S. Rajbhandari et al. [5] describe methods for reducing this redundancy by partitioning the (1) optimizer state, (2) gradients, and (3) parameters across ranks.

In this part, we'll implement a simplified version of optimizer state sharding. Each rank's optimizer instance handles only a subset of the parameters (approximately $1 / \text{world\_size}$). After each optimizer step, each rank broadcasts its updated parameters to the other ranks.

**Problem (optimizer_state_sharding): Optimizer State Sharding (15 points)**

Implement a Python class to handle optimizer state sharding. Public interface:

- `def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any)`
- `def step(self, closure, **kwargs)`
- `def add_param_group(self, param_group: dict[str, Any])`

**Deliverable:** Implement the container class. Implement the adapter `adapters.get_sharded_optimizer`. Run `uv run pytest tests/test_sharded_optimizer.py`.

**Problem (optimizer_state_sharding_accounting): Optimizer State Sharding Accounting (5 points)**

(a) Profile peak memory usage with and without optimizer state sharding. Use 1 node, 2 GPUs, `xl` model size. Report peak memory after model initialization, before optimizer step, and after optimizer step.

**Deliverable:** 2-3 sentence response with peak memory usage results and breakdown.

(b) Measure the time taken per iteration with and without optimizer state sharding.

**Deliverable:** 2-3 sentence response with your timings.

(c) How does our approach differ from ZeRO stage 1 (ZeRO-DP $P_{os}$ in [5])?

**Deliverable:** 2-3 sentence summary.

---

## 7 Fully-Sharded Data Parallel

With optimizer state sharding and data parallel, we can split optimizer state and activations. However, model weights remain duplicated. FSDP solves this: each GPU stores only its own slice of every weight tensor, and pulls slices from other GPUs via all-gather for forward/backward passes.

**Problem (fsdp): Fully-Sharded Data Parallel (15 points)**

Implement a Python class for fully-sharded data parallel training. Public interface:

- `def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None)`
- `def forward(self, *inputs, **kwargs)`
- `def finish_gradient_synchronization(self)`

**Deliverable:** Implement the container class. Implement the adapter `adapters.get_fsdp`. Run `uv run pytest tests/test_fsdp.py`.

**Problem (fsdp_accounting): FSDP Accounting (5 points)**

(a) Given your analysis in Section 6, how much memory do you expect to save from the peak by implementing FSDP?

**Deliverable:** 2-3 sentence response with your findings.

(b) Profile the `xl` model on two GPUs and pay attention to the all-gather of weights. Does the communication finish in time for the forward pass?

**Deliverable:** 2-3 sentence response with your timings. Include screenshots of Nsight.


---

## 8 Analyzing Parallelism Strategies

Common parallelism strategies include:

- **Data parallelism (DP)** — Batches split across devices, gradients averaged.
- **Fully-Sharded Data Parallelism (FSDP)** — Also splits optimizer states, gradients, and weights.
- **Tensor Parallelism (TP)** — Weight matrices sharded across input/output dimension.
- **Pipeline Parallelism (PP)** — Model split layerwise into stages on different devices.
- **Expert Parallelism (EP)** — Experts in MoE models split onto different devices.

### 8.1 Communication Primitives

Suppose we have $N$ devices, each with $W$ egress bandwidth. A ring all-gather takes $\frac{N-1}{N} \cdot \frac{S}{W}$ seconds. A ring reduce-scatter takes $\frac{N-1}{N} \cdot \frac{S}{W}$ seconds. A ring all-reduce (reduce-scatter + all-gather) takes $2 \cdot \frac{N-1}{N} \cdot \frac{S}{W}$ seconds.

**Problem (alternate_ring_all_reduce): Alternate ring all-reduce (1 point)**

An alternative algorithm is described. How long does it take?

**Deliverable:** An answer in terms of $S$, $N$, and $W$, with one-sentence justification.

### 8.2 Analyzing Data Parallel

Forward pass of a single FFN layer:

$$
x_1 = x W_1 \tag{20}
$$
$$
x_2 = x W_2 \tag{21}
$$
$$
z = f(x_1) * x_2 \tag{22}
$$
$$
y = z W_3 \tag{23}
$$

Backward pass:

$$
dz = dy W_3^\top \tag{24}
$$
$$
dx_2 = dz * f(x_1) \tag{25}
$$
$$
dx_1 = dz * f'(x_1) * x_2 \tag{26}
$$
$$
dx = dx_1 W_1^\top + dx_2 W_2^\top \tag{27}
$$
$$
dW_3 = z^\top dy \tag{28}
$$
$$
dW_2 = x^\top dx_2 \tag{29}
$$
$$
dW_1 = x^\top dx_1 \tag{30}
$$

With $N_{DP}$ devices, shard input $x$ into shards of size $(B / N_{DP}, D)$. Gradients are:

$$
dW_3^{(i)} = z^{(i)\top} dy^{(i)} \tag{31}
$$
$$
dW_2^{(i)} = x^{(i)\top} dx_2^{(i)} \tag{32}
$$
$$
dW_1^{(i)} = x^{(i)\top} dx_1^{(i)} \tag{33}
$$

An all-reduce is needed to get full gradients.

**Problem (data_parallel_calcs): Data parallel calculations (3 points)**

(a) How many FLOPs for backward pass with $N_{DP}$ DP? (Ignore non-matmul ops.)

**Deliverable:** An answer in terms of $B$, $D$, $D_{FF}$, $N_{DP}$, with justification.

(b) How much communication time in backward pass with $N_{DP}$ DP?

**Deliverable:** An answer in terms of $B$, $D$, $D_{FF}$, $N_{DP}$, $W$, with justification.

(c) How large can $N_{DP}$ become before communication bottlenecked?

**Deliverable:** An inequality with $N_{DP}$, in terms of $B$, $D$, $D_{FF}$, $C$, $W$, with justification.

### 8.3 Analyzing Fully Sharded Data Parallel

With FSDP, weights are also sharded. Forward pass requires all-gather on weights first. Backward pass requires all-gather then reduce-scatter on gradients.

$$
W_1 = \text{all-gather}(\{W_1^{(i)}\}) \tag{34}
$$
$$
W_2 = \text{all-gather}(\{W_2^{(i)}\}) \tag{35}
$$
$$
W_3 = \text{all-gather}(\{W_3^{(i)}\}) \tag{36}
$$

$$
dW_1^{(i)} = \text{reduce-scatter}(\{dW_1^{(i)}\}) \tag{42}
$$
$$
dW_2^{(i)} = \text{reduce-scatter}(\{dW_2^{(i)}\}) \tag{43}
$$
$$
dW_3^{(i)} = \text{reduce-scatter}(\{dW_3^{(i)}\}) \tag{44}
$$

**Problem (fsdp_calcs): Fully sharded data parallel calculations (3 points)**

(a) How many FLOPs for forward and backward pass with $N_{FSDP}$?

**Deliverable:** Two answers in terms of $B$, $D$, $D_{FF}$, $N_{FSDP}$, with justifications.

(b) How much communication time in forward and backward pass?

**Deliverable:** Two answers in terms of $B$, $D$, $D_{FF}$, $N_{FSDP}$, $W$, with justifications.

(c) How large can $N_{FSDP}$ become before communication bottlenecked?

**Deliverable:** Two inequalities with $N_{FSDP}$, with justifications.

### 8.4 Analyzing Tensor Parallel

In TP, weight matrices are sharded. Column parallel: $xW = \text{all-gather}(\{x W^{(i)}\})$. Row parallel: $xW = \text{all-reduce}(\{x^{(i)} W^{(i)}\})$.

For our FFN, $W_1$ and $W_2$ are column parallel, $W_3$ is row parallel:

$$
x_1^{(i)} = x W_1^{(i)} \tag{47}
$$
$$
x_2^{(i)} = x W_2^{(i)} \tag{48}
$$
$$
z^{(i)} = f(x_1^{(i)}) * x_2^{(i)} \tag{49}
$$
$$
y^{(i)} = z^{(i)} W_3^{(i)} \tag{50}
$$
$$
y = \text{all-reduce}(\{y^{(i)}\}) \tag{51}
$$

**Problem (tp_calcs): Tensor parallel calculations (4 points)**

(a) Write out the backward pass of the TP strategy.

(b) How many FLOPs for forward and backward pass with $N_{TP}$ TP?

(c) How much communication time in forward and backward pass?

(d) How large can $N_{TP}$ become before communication bottlenecked?

### 8.5 2D Parallelism (FSDP + TP)

Combine FSDP and TP into a 2D grid with $N = N_{TP} N_{FSDP}$ devices.

Forward pass with batch-sharded input $x^{(j)}$ of size $(\frac{B}{N_{FSDP}}, D)$:

$$
W_1^{(i)} = \text{all-gather}(\{W_1^{(i,j)}\}) \tag{52}
$$
$$
W_2^{(i)} = \text{all-gather}(\{W_2^{(i,j)}\}) \tag{53}
$$
$$
W_3^{(i)} = \text{all-gather}(\{W_3^{(i,j)}\}) \tag{54}
$$
$$
x_1^{(i,j)} = x^{(j)} W_1^{(i)} \tag{55}
$$
$$
x_2^{(i,j)} = x^{(j)} W_2^{(i)} \tag{56}
$$
$$
z^{(i,j)} = f(x_1^{(i,j)}) * x_2^{(i,j)} \tag{57}
$$
$$
y^{(i,j)} = z^{(i,j)} W_3^{(i)} \tag{58}
$$
$$
y^{(j)} = \text{all-reduce}(\{y^{(i,j)}\}) \tag{59}
$$

**Problem (fsdp_tp_calcs): 2D parallelism calculations (6 points)**

(a) How many FLOPs for forward pass with $N_{FSDP}$ FSDP + $N_{TP}$ TP?

(b) How much communication time in forward pass? Assume FSDP and TP collectives can be overlapped.

(c) Under optimal $N_{TP}$ and $N_{FSDP}$, how large can $N = N_{TP} N_{FSDP}$ become before communication bottlenecked?

(d) Now suppose FSDP-axis and TP-axis collectives cannot be overlapped. How large can $N$ become?


---

## 9 Leaderboard

Assignment 2's leaderboard will test the speed of a full training step for an 8B model. The key restriction is that you cannot change the input/output behavior of the model.

```python
class Config:
    ctx_len = 32768
    vocab_size = 151936
    d_model = 4096
    d_ff = 11008
    num_layers = 34
    num_heads = 32
    torch_dtype = torch.bfloat16
    is_causal = True
    batch_size = 2
```

**Ideas for improvement:**

- Tune tile sizes for your kernel (use Triton autotune)
- Tune additional Triton/torch.compile config parameters
- Implement fused AdamW
- Fuse LM head and cross-entropy loss
- Improve FlashAttention:
  - Implement backward pass in Triton
  - Two passes for backward (dQ and dK/dV)
  - Early termination for causal masking
  - Use TMA on Hopper+
- Use activation checkpointing if needed

**Problem (leaderboard): Leaderboard: fastest training step (10 points)**

The benchmark will be run at batch size 2 on two B200 GPUs. Your submission will be evaluated on wall-clock time for a complete training step (forward pass, loss, backward pass, and AdamW update).

**Deliverable:** Your best wall-clock time for a full forward-and-backward training step with AdamW.

Submit your result to the leaderboard here: <https://github.com/stanford-cs336/assignment2-systems-leaderboard>

---

## Bibliography

[1] T. Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." [Online]. Available: <https://arxiv.org/abs/2307.08691>

[2] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Re, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," in *Advances in Neural Information Processing Systems*, 2022. [Online]. Available: <https://openreview.net/forum?id=H4DqfPSibmx>

[3] M. Milakov and N. Gimelshein, "Online normalizer calculation for softmax." [Online]. Available: <https://arxiv.org/abs/1805.02867>

[4] H. He, "Making Deep Learning Go Brrrr From First Principles," 2022. [Online]. Available: <https://horace.io/brrr_intro.html>

[5] S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He, "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models," 2020.

[6] J. Austin et al., "How to Scale Your Model," 2025.

[7] H. Z. P. N. M. M. L. W. T. W. Nouamane Tazi Ferdinand Mom, "The Ultra-Scale Playbook: Training LLMs on GPU Clusters," 2025.
