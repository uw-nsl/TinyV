# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# setup.py is the fallback installation script when pyproject.toml does not work
from setuptools import setup, find_packages
import os

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, 'verl/version/version')) as f:
    __version__ = f.read().strip()

install_requires = [
  'accelerate',
  'codetiming',
  'datasets',
  'dill',
  'hydra-core',
  'numpy',
  'pandas',
  'peft',
  'pyarrow>=15.0.0',
  'pybind11',
  'pylatexenc',
  'ray[default]>=2.10',
  'tensordict<=0.6.2',
  'torchdata',
  'transformers',
  'wandb',
  'mathruler',
  'vllm==0.8.5',
]

TEST_REQUIRES = ['pytest', 'yapf', 'py-spy']
PRIME_REQUIRES = ['pyext']
GEO_REQUIRES = ['mathruler']
GPU_REQUIRES = ['liger-kernel', 'flash-attn']
MATH_REQUIRES = ['math-verify']  # Add math-verify as an optional dependency
VLLM_REQUIRES = ['tensordict<=0.6.2', 'vllm<=0.8.2']
SGLANG_REQUIRES = [
  'tensordict<=0.6.2', 
  'sglang[all]==0.4.4.post3',
  'torch-memory-saver>=0.0.5'
]

extras_require = {
  'test': TEST_REQUIRES,
  'prime': PRIME_REQUIRES,
  'geo': GEO_REQUIRES,
  'gpu': GPU_REQUIRES,
  'math': MATH_REQUIRES,
  'vllm': VLLM_REQUIRES,
  'sglang': SGLANG_REQUIRES,
}

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='verl',
    version=__version__,
    package_dir={'': '.'},
    packages=find_packages(where='.'),
    url='https://github.com/volcengine/verl',
    license='Apache 2.0',
    author='Bytedance - Seed - MLSys',
    author_email='zhangchi.usc1992@bytedance.com, gmsheng@connect.hku.hk',
    description='verl: Volcano Engine Reinforcement Learning for LLM',
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={'': ['version/*'],
                  'verl': ['trainer/config/*.yaml'],},
    include_package_data=True,
    long_description=long_description,
    long_description_content_type='text/markdown'
)