# Running DeepSeek-V3-0324 (685 B parameters) with 512 GB RAM on a single 16 GB GPU

You can watch the video here! [Running DeepSeek-V3-0324 with 512 GB RAM on a single 16 GB GPU](https://www.youtube.com/watch?v=bM9tAYXK8dE)

Here's my guide in how I set up `ktransformers` (quick transformers) to perform inference on **DeepSeek-V3-0324** with a 20c/40t CPU, 512 GB DDR5 RAM and just a **single** Nvidia A4000 16 GB GPU. 

I think there needs to be a proper guide to use `ktransformers` with large MoE models. A lot of resources (especially YouTube) will show how to run these very large models on other inference engines like Ollama, or llama.cpp. There are some interesting build guides too, like running DeepSeek on an Epyc CPU with just lots of RAM. 

But one video caught my attention and that's [Jesse's (createthis) video](https://www.youtube.com/watch?v=fI6uGPcxDbM) on his setup where he has a 2x EPYC CPU setup with 768 GB RAM and just a single 24 GB (RTX 3090) GPU. After seeing him using `ktransformers` and the inferencing speed he achieved with it, I decided I had to do the same with my system.

My system isn't as powerful, but it gets close to the performance that Jesse showed in his video from the eye test.

My desktop workstation setup:
* Intel Xeon w5-3535x (20 cores/40 threads)
* 512 GB DDR5 RDIMM at 5600 MT/s (64 GB x 8 Samsung DIMMs)
* NVIDIA A4000 16 GB GPU (Ampere generation)
* 4.0 TB Crucial SSD to store the model

All in total, you can purchase this same system for less than an Mac Studio M3 Ultra, or a single RTX 6000 Ada GPU.

But notice the GPU that I have. It's just an older Ampere NVIDIA GPU with just 16 GB of VRAM. This is what I consider to be amazing about all of this. Running big MoE models is usable with the ktransformers approach.

# ktransformers

This is a game changer. I would say that I can't live without it now. But it isn't easy to set up. You can follow the official guide here, but it was a bit tough to work out all the gotchas. 

Your life will be much easier if you have CUDA properly installed. That means:

* Recent NVIDIA drivers (mine was 570)
* Latest NVIDIA CUDA toolkit (12.8)

You will also have a good time if you are on at least Ubuntu 22.04.5 or 24.04.2. My system is on Ubuntu 22.04.5. 

Anyway, here is what we need to do:

## Install Miniconda and Create a Conda Environment

I haven't tried this outside of `conda` so I will just be safe here and just use it for this guide. 

Download and run the Miniconda script to set it up:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

Then you can activate miniconda at any time by invoking:

```bash
source ~/miniconda3/bin/activate
```

## Install Dependencies and Create Conda Environment

```
sudo apt update && sudo apt install build-essential cmake ninja-built patchelf
```

Then create the conda environment and also make sure you have the latest `libstdc++`.

```
conda create --name ktransformers python=3.11
conda activate ktransformers

conda install -c conda-forge libstdcxx-ng
```

You can validate the install of `libstdc++` if you run the following command and see that you have the right version: 3.4.32

```
strings $HOME/miniconda3/envs/ktransformers/lib/libstdc++.so.6 | grep GLIBCXX
```

Now install all the python dependencies:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip3 install packaging ninja cpufeature numpy
```

Recommended that you really use the URL provided. You don't want to end up in a bad state with the wrong PyTorch dependencies.

## Install the Appropriate Flash Attention

Here's how to determine which version you need:

* Choose the right cuda version (12)
* Choose the right PyTorch version (2.6)
* ABI = True

Download the `whl` file and just execute with:

```
pip3 install flash_attention_blah_blah.whl
```

## Clone and Build the Repo

RECOMMENDED! Check out a tagged release! 

```
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
git checkout v0.0.0 # CHECK OUT A TAGGED RELEASE
git submodule update --init --recursive
```

Then build. Currently as of 0.2.4post1, something is broken, and the `balanced_server` backend doesn't work. You'll get something like `sched_ext` mising upon running. Note, even if you provide `ktransformers` as the backend, you'll still get this error. If you run into it, don't worry, we'll address it in the next section: **sched_text Fix**.

```
USE_BALANCE_SERVE=0 bash ./install.sh
```

Takes a while to build.

## sched_ext Fix

If you have issues with `sched_ext` you can follow this to get fixed.

Then follow this guide to fix the `sched_ext` stuff: https://github.com/kvcache-ai/ktransformers/issues/1017#issuecomment-2778734503 

Edit the file: `$HOME/miniconda3/envs/ktransformers/lib/python3.11/site-packages/ktransformers/server/balance_serve/settings.py` . Make these changes:

```python
import sched_ext

# becomes
try:
  import sched_ext
except ImportError:
  sched_ext = None
```

Edit the file: `$HOME/miniconda3/envs/ktransformers/lib/python3.11/site-packages/ktransformers/models/custom_cache.py`. Make these changes:

```python
from ktransformers.server.balance_serve.settings import sched_ext

# becomes
try:
	from ktransformers.server.balance_serve.settings import sched_ext
except ImportError:
  class DummyModule:
    class InferenceContext:
      pass
  sched_ext = DummyModule()
```

Then remove type hints in `KDeepSeekV3Cache` class:

```python
# Changed from:
def load(self, inference_context: sched_ext.InferenceContext):
# To:
def load(self, inference_context):
```

## Prepare Your Models

Create a `DeepSeek-V3-0324-Config` folder that contains the configuration files of the model. This is just basically the HuggingFace repo without the `safetensors` file. Here's a script to use `wget` to download them. I modified this from [createthis larry document](https://github.com/createthis/larry):

```
mkdir $HOME/DeepSeek-V3-0324-Config
cd $HOME/DeepSeek-V3-0324-Config

wget https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/raw/main/LICENSE
wget https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/raw/main/README.md
wget https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/raw/main/config.json
wget https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/raw/main/configuration_deepseek.py
wget https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/raw/main/model.safetensors.index.json
wget https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/raw/main/modeling_deepseek.py
wget https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/raw/main/tokenizer.json
wget https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/raw/main/tokenizer_config.json

cd -
```

Then add the GGUF in a separate folder:

```
mkdir $HOME/DeepSeek-V3-0324-Q4_K_M
mv $HOME/DeepSeek-V3-0324-Q4_K_M.gguf $HOME/DeepSeek-V3-0324-Q4_K_M
```

## Running!

Make sure you have lots of free RAM to run this. DeepSeek is super big. 

Run the thing!

```
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST="8.6"

python ktransformers/server/main.py \
	--host 0.0.0.0 \
	--port 9999 \
	--model_path $HOME/DeepSeek-V3-0324-Config \
	--model_name "DeepSeek-V3-0324-Q4_K_M" \
	--gguf_path $HOME/DeepSeek-V3-0324-Q4_K_M \
	--chunk_size 256 \
	--cache_lens 16384 \
	--cpu_infer 38 \
	--max_new_tokens 4096 \
	--temperature 0.3 \
	--backend_type ktransformers
```

To hook this in Open Web UI or other UI inference things, you just add an OpenAI API connection. For example:

```
http://localhost:9999/v1
```

Should be like any other connection now. 
