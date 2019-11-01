# Deep Video Deblurring: The Devil is in the Details

--------------------------------------------------------------------------------

This PyTorch implementation accompanies the paper:

Jochen Gast and Stefan Roth, **Deep Video Deblurring: The Devil is in the Details**,<br>
ICCV Workshop on Learning for Computational Imaging (ICCVW), Seoul, Korea, November, 2019

Preprint: https://arxiv.org/abs/1909.12196 <a href="https://www.visinf.tu-darmstadt.de"> <img align="right" src="https://github.com/ezjong/deepvisinf/blob/deblur-devil-release/resources/vi-logo-small.jpg" alt="DeepVisinf Logo" height="55"/> </a> <br>
Contact: Jochen Gast (<jochen.gast@visinf.tu-darmstadt.de>) <p align="right"> <a href="https://www.visinf.tu-darmstadt.de">Visual Inference</a> </p>

--------------------------------------------------------------------------------

- [Components for Deep Video Deblurring](#components-for-deep-video-deblurring)
- [Datasets](#datasets)
- [Installation, External Packages & Telegram Bot](#installation-external-packages--telegram-bot)
- [Deep Visual Inference Framework](#deep-visual-inference-framework)
  - [Adding Classes](#adding-classes)
  - [Everything is a Dictionary](#everything-is-a-dictionary)
  - [Magic Keys](#magic-keys)
  - [Data Loader Collation](#data-loader-collation)
  - [Logging Facilities](#logging-facilities)
  - [Custom Tensorboard Calls](#custom-tensorboard-calls)
  - [More Settings](#more-settings)
- [License](#license)
- [Citation](#citation)

--------------------------------------------------------------------------------

## Components for Deep Video Deblurring
Ignoring the framework code, the components for Deep Video Deblurring are contained in the modules

| Modules | Description |
| ---- | --- |
| [**datasets.gopro_nah**](datasets/gopro_nah.py) | implementation (+ augmentations) for reading Nah et al's GoPro dataset |
| [**datasets.gopro_su**](datasets/gopro_su.py) | implementation (+ augmentations) for reading Su et al's GoPro dataset |
| [**losses.deblurring_error**](losses/deblurring_error.py) | loss implementation (just plain MSE) |
| [**models.deblurring.dbn**](models/deblurring/dbn.py) | our baseline implementation for Su et al's DBN network |
| [**models.deblurring.flowdbn**](models/deblurring/flowdbn.py) | our FlowDBN implementation  |
| [**models.flow.flownet1s**](models/flow/flownet1s.py) | FlowNet1S optical flow network used in FlowDBN |
| [**models.flow.pwcnet**](models/flow/pwcnet.py) | PWC optical flow network used in FlowDBN |
| [**visualizers.gopro_inference**](visualizers/gopro_inference.py) | TensorBoard Visualizer for inference (on a given dataset) |
| [**visualizers.gopro_visualizer**](visualizers/gopro_visualizer.py) | TensorBoard Visualizer for monitoring training progress |


## Datasets
The mirrors of the GoPro datasets can be found here:

| GoPro Dataset | Paper |
| ---- | --- | 
| [**Su et al**](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/DeepVideoDeblurring_Dataset.zip)<br> [mirror](https://dl.dropboxusercontent.com/s/fxied15l3xlxm79/DeepVideoDeblurring_Dataset.zip) | S. Su, M. Delbracio, J. Wang, G. Sapiro, W. Heidrich, and O. Wang.<br>[Deep video deblurring for hand-held cameras](https://github.com/shuochsu/DeepVideoDeblurring), CVPR, 2017. |
| [**Nah et al**](https://drive.google.com/file/d/1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2/view?usp=sharing)<br> [mirror](https://dl.dropboxusercontent.com/s/rwlliuyk513bwum/GOPRO_Large.zip) | S. Nah, T. H. Kim, and K. M. Lee.<br>[Deep multi-scale convolutional neural network for dynamic scene deblurring](https://github.com/SeungjunNah/DeepDeblur_release), CVPR, 2017. |


## Installation, External Packages & Telegram Bot

This code has been tested with Python 3.6, PyTorch 1.2, Cuda 10.0 but is likely to run with newer versions of PyTorch and Cuda. For installing PyTorch, follow the steps [here](https://pytorch.org/get-started/locally/). 

Additional dependencies can be installed by running the script
```bash
pip install -r requirements.txt
```

The code for the PWC network relies on a correlation layer implementation provided by [Clément Pinard](https://github.com/ClementPinard/Pytorch-Correlation-extension).<br>
To install the correlation layer, run
```bash
cd external_packages && . install_spatial_correlation_sampler.sh
```

The framework supports status updates sent via Telegram, i.e. using
```python
logging.telegram(msg, *args)
```
Per default, any finished epoch will send an update with the current losses.  Note that the dependencies for the Telegram bot, i.e. ``filetype jsonpickle telegrambotapiwrapper``, are included in ``requirements.txt``, but they are not essential to the framework.

In order to use the Telegram bot, you need to copy ``dot_machines.json`` to ``~/.machines.json`` and fill in the required ``chat_id`` and ``tokens``. [Here](https://core.telegram.org/bots#6-botfather) you can find out how to obtain these credentials.

## Scripts and Models

The scripts make use of the environment variables ``DATASETS_HOME`` pointing to the root folder of the respective dataset, 
and ``EXPERIMENTS_HOME`` to create output artifacts.

| Scripts | Description |
| ---- | --- |
| [**scripts.deblurring.gopro_nah**](scripts/deblurring/gopro_nah) | Scripts for GoPro by Nah et al. |
| [**scripts.deblurring.gopro_su**](scripts/deblurring/gopro_su) | Scripts for GoPro by Su et al. |

To train the models you need the prewarping networks. Place them in ``$EXPERIMENTS_HOME/prewarping_networks``.

| Models | Description |
| ---- | --- |
| [**Prewarping Networks**](https://dl.dropboxusercontent.com/s/h9k5oasjoayyfh1/prewarping_networks.zip) | FlowNet1S and PWCNet |

More models will follow soon!


## Deep Visual Inference Framework

### Adding Classes
For adding new classes/functions following packages are relevant:

| Modules | Description |
| ---- | --- |
| [**augmentations**](augmentations/) | Register (gpu) augmentations |
| [**contrib**](contrib/) | Supposed to be used for PyTorch utilities |
| [**datasets**](datasets/) | Register datasets |
| [**losses**](losses/) | Register losses |
| [**models**](models/) | Register models  |
| [**optim**](optim/) | Register optimizers |
| [**utils**](utils/) | Any non-PyTorch related utilities |
| [**visualizers**](visualizers/) | Register Tensorboard visualizers |

### Everything is a Dictionary 
Most things are automated for convenience. To accomodate this automation, there are three essential Python dictionaries being passed around which hold any artifact produced along the way

| Dictionary Name | Description |
| ---- | --- |
| **example_dict**| The example dictionary. Gets created by a dataset and is supposed to hold all data relevant to a dataset example. Note that an ``augmentation`` is supposed to receive and return an ``example_dict``. The example dictionary is received by ``augmentations``, ``losses``, ``models``, and ``visualizers``.
| **model_dict** | The model dictionary. Gets created by a model. Supposed to hold any data, i.e. predictions, generated by a model. The example dictionary is received by ``losses`` and ``visualizers``. |
| **loss_dict** | The loss dictionary. Gets created by a loss. Supposed to hold any data, i.e. metrics, generated by a loss. The loss dictionary is received by ``visualizers``. |

Keep this in mind, when adding new functionality.

### Magic Keys
Again, the framework passes around dictionaries between augmentations, datasets, losses, models, and visualizers. You may want a subset of these tensors to be transfered to Cuda along the way. The are two magic prefixes for dictionary keys to assure that, i.e. ``input*`` and ``target*``. Any input or target tensor that should be transfered to Cuda along the way, has to start with one of these prefixes.

E.g. a dataset may return a dictionary with the keys ``input1`` and ``target1`` (both cpu tensors); any augmentation, loss, or model will fetch these as gpu tensors.

### Data Loader Collation
This is in particular interesting for the GoPro datasets. The framework allows dataloaders to return multiple -- typically photometric -- augmentations per single example, in order to increase throughput when loading large images. In such a case a tensor returned from a single dataset example should have three dimensions. The pipeline will stack the first dimension into a batch.

Let's say, a dataset returns four examples per load, i.e. ``input1`` and ``target1`` tensors have the dimension ``4xHxW``. For, e.g. ``batch_size=8``, the resulting tensors (received by the augmentation, loss, model, and visualizers) will be ``32xHxW``.

### Logging Facilities
There are some automated logging facilities which log any output into the given ``save`` directory.
- A ``args.txt`` and ``args.json`` file remembering the passed arguments.
- A ``logbook.txt`` gets created remembering all calls to ``logging.info(msg, *args)``.
- CSV/MAT file generation of any loss (of any epoch) found in the ``loss_dict``.
- For every run, the complete source code gets zipped into the log directory.
- There is always a tensorboard created at the ``$SAVE/tb`` directory.

### Custom Tensorboard Calls
When calling tensorboard summaries in your models for debugging, I suggest to replace any call to ``torch.utils.tensorboard.SummaryWriter.add_*`` by ``utils.summary.*``

For instance, instead of
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(..)
writer.add_scalar(tensor1)
writer.add_images(tensor2)
```
use
```python
from utils import summary
summary.scalar(tensor1)
summary.images(tensor2)
```

Advantages:
- This will automatically use the created tensorboard in the ``save`` directory.
- The current epoch gets automatically set by the framework.
- Otherwise, they are identical.

### More Settings
A limited amount of customization is available in ``constants.py`` where you can set
- Logging indents, time zones, and colors
- Filename defaults
- Flush intervals
- etc


## License
This code is licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0), as found in the LICENSE file.


## Citation

```bibtex
# brief
@inproceedings{Gast:2019:DVD,
    Author = {Jochen Gast and Stefan Roth},
    Booktitle = {ICCV Workshops},
    Title = {Deep Video Deblurring: {T}he Devil is in the Details},
    Year = {2019}
}
```

```bibtex
# detailed
@inproceedings{Gast:2019:DVD,
    Address = {Seoul, Korea},
    Author = {Jochen Gast and Stefan Roth},
    Booktitle = {ICCV Workshop on Learning for Computational Imaging (ICCVW)},
    Month = nov,
    Title = {Deep Video Deblurring: {T}he Devil is in the Details},
    Year = {2019}
}
```
