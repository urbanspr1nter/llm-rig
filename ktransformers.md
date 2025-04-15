# Setting up ktransformers to run DeepSeek-V3-0324 FAST

Here's my guide in how I set up ktransformers to make DeepSeek-V3-0324 run fast on CPU inference! 





# ktransformers

This is a freaking game changer. I would say that I cannot live without it now. But it isn't easy to set up.

Here is what we need to do:

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

For this one we use CUDA 12.6



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

Then build. Currently as of 0.2.4post1, something is broken, and the balanced_server backend doesn't work. You'll get something like `sched_ext` mising upon running.

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

Create a `DeepSeek-V3-0324-Config` folder that contains the configuration files of the model. This is just basically the HuggingFace repo without the `safetensors` file. Here's a script to use `wget` to download them:

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
mkdir $HOME/DeepSeek-V3-0324-Q5_K_M
mv $HOME/DeepSeek-V3-0324-Q5_K_M.gguf $HOME/DeepSeek-V3-0324-Q5_K_M
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
	--model_name "DeepSeek-V3-0324-Q5_K_M" \
	--gguf_path $HOME/DeepSeek-V3-0324-Q5_K_M \
	--chunk_size 256 \
	--cache_lens 131072 \
	--cpu_infer 48 \
	--max_new_tokens 8192 \
	--temperature 0.3 \
	--backend_type ktransformers
```

To hook this in Open Web UI just add an OpenAI API connection. For example:

```
http://localhost:9999/v1
```

Should be like any other connection now. 