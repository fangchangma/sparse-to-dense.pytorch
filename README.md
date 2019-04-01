sparse-to-dense.pytorch
============================

This repo implements the training and testing of deep regression neural networks for ["Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image"](https://arxiv.org/pdf/1709.07492.pdf) by [Fangchang Ma](http://www.mit.edu/~fcma) and [Sertac Karaman](http://karaman.mit.edu/) at MIT. A video demonstration is available on [YouTube](https://youtu.be/vNIIT_M7x7Y).
<p align="center">
	<img src="http://www.mit.edu/~fcma/images/ICRA2018.png" alt="photo not available" width="50%" height="50%">
	<img src="https://j.gifs.com/Z4qDow.gif" alt="photo not available" height="50%">
</p>

This repo can be used for training and testing of
- RGB (or grayscale image) based depth prediction
- sparse depth based depth prediction
- RGBd (i.e., both RGB and sparse depth) based depth prediction

The original Torch implementation of the paper can be found [here](https://github.com/fangchangma/sparse-to-dense).

## Contents
0. [Requirements](#requirements)
0. [Training](#training)
0. [Testing](#testing)
0. [Trained Models](#trained-models)
0. [Benchmark](#benchmark)
0. [Citation](#citation)

## Requirements
This code was tested with Python 3 and PyTorch 0.4.0.
- Install [PyTorch](http://pytorch.org/) on a machine with CUDA GPU.
- Install the [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) and other dependencies (files in our pre-processed datasets are in HDF5 formats).
	```bash
	sudo apt-get update
	sudo apt-get install -y libhdf5-serial-dev hdf5-tools
	pip3 install h5py matplotlib imageio scikit-image opencv-python
	```
- Download the preprocessed [NYU Depth V2](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and/or [KITTI Odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) dataset in HDF5 formats, and place them under the `data` folder. The downloading process might take an hour or so. The NYU dataset requires 32G of storage space, and KITTI requires 81G.
	```bash
	mkdir data; cd data
	wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz
	tar -xvf nyudepthv2.tar.gz && rm -f nyudepthv2.tar.gz
	wget http://datasets.lids.mit.edu/sparse-to-dense/data/kitti.tar.gz
 	tar -xvf kitti.tar.gz && rm -f kitti.tar.gz
	cd ..
	```
## Training
The training scripts come with several options, which can be listed with the `--help` flag. 
```bash
python3 main.py --help
```

For instance, run the following command to train a network with ResNet50 as the encoder, deconvolutions of kernel size 3 as the decoder, and both RGB and 100 random sparse depth samples as the input to the network.
```bash
python3 main.py -a resnet50 -d deconv3 -m rgbd -s 100 --data nyudepthv2
```

Training results will be saved under the `results` folder. To resume a previous training, run
```bash
python3 main.py --resume [path_to_previous_model]
```

## Testing
To test the performance of a trained model without training, simply run main.py with the `-e` option. For instance,
```bash
python3 main.py --evaluate [path_to_trained_model]
```

## Trained Models
A number of trained models is available [here](http://datasets.lids.mit.edu/sparse-to-dense.pytorch/results/).

## Benchmark
The following numbers are from the original Torch repo.
- Error metrics on NYU Depth v2:

	| RGB     |  rms  |  rel  | delta1 | delta2 | delta3 |
	|-----------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|
	| [Roy & Todorovic](http://web.engr.oregonstate.edu/~sinisa/research/publications/cvpr16_NRF.pdf) (_CVPR 2016_) | 0.744 | 0.187 |  - | - | - |
	| [Eigen & Fergus](http://cs.nyu.edu/~deigen/dnl/) (_ICCV 2015_)  | 0.641 | 0.158 | 76.9 | 95.0 | 98.8 |
	| [Laina et al](https://arxiv.org/pdf/1606.00373.pdf) (_3DV 2016_)            | 0.573 | **0.127** | **81.1** | 95.3 | 98.8 |
	| Ours-RGB             | **0.514** | 0.143 | 81.0 | **95.9** | **98.9** |

	| RGBd-#samples   |  rms  |  rel  | delta1 | delta2 | delta3 |
	|-----------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|
	| [Liao et al](https://arxiv.org/abs/1611.02174) (_ICRA 2017_)-225 | 0.442 | 0.104 | 87.8 | 96.4 | 98.9 |
	| Ours-20 | 0.351 | 0.078 | 92.8 | 98.4 | 99.6 |
	| Ours-50 | 0.281 | 0.059 | 95.5 | 99.0 | 99.7 |
	| Ours-200| **0.230** | **0.044** | **97.1** | **99.4** | **99.8** |

	<img src="http://www.mit.edu/~fcma/images/ICRA18/acc_vs_samples_nyu.png" alt="photo not available" width="50%" height="50%">

- Error metrics on KITTI dataset:

	| RGB     |  rms  |  rel  | delta1 | delta2 | delta3 |
	|-----------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|
	| [Make3D](http://papers.nips.cc/paper/5539-depth-map-prediction-from-a-single-image-using-a-multi-scale-deep-network.pdf) | 8.734 | 0.280 | 60.1 | 82.0 | 92.6 |
	| [Mancini et al](https://arxiv.org/pdf/1607.06349.pdf) (_IROS 2016_)  | 7.508 | - | 31.8 | 61.7 | 81.3 |
	| [Eigen et al](http://papers.nips.cc/paper/5539-depth-map-prediction-from-a-single-image-using-a-multi-scale-deep-network.pdf) (_NIPS 2014_)  | 7.156 | **0.190** | **69.2** | 89.9 | **96.7** |
	| Ours-RGB             | **6.266** | 0.208 | 59.1 | **90.0** | 96.2 |

	| RGBd-#samples   |  rms  |  rel  | delta1 | delta2 | delta3 |
	|-----------------------------|:-----:|:-----:|:-----:|:-----:|:-----:|
	| [Cadena et al](https://pdfs.semanticscholar.org/18d5/f0747a23706a344f1d15b032ea22795324fa.pdf) (_RSS 2016_)-650 | 7.14 | 0.179 | 70.9 | 88.8 | 95.6 |
	| Ours-50 | 4.884 | 0.109 | 87.1 | 95.2 | 97.9 |
	| [Liao et al](https://arxiv.org/abs/1611.02174) (_ICRA 2017_)-225 | 4.50 | 0.113 | 87.4 | 96.0 | 98.4 |
	| Ours-100 | 4.303 | 0.095 | 90.0 | 96.3 | 98.3 |
	| Ours-200 | 3.851 | 0.083 | 91.9 | 97.0 | 98.6 |
	| Ours-500| **3.378** | **0.073** | **93.5** | **97.6** | **98.9** |

	<img src="http://www.mit.edu/~fcma/images/ICRA18/acc_vs_samples_kitti.png" alt="photo not available" width="50%" height="50%">

	Note: our networks are trained on the KITTI odometry dataset, using only sparse labels from laser measurements.

## Citation
If you use our code or method in your work, please consider citing the following:

	@article{Ma2017SparseToDense,
		title={Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image},
		author={Ma, Fangchang and Karaman, Sertac},
		booktitle={ICRA},
		year={2018}
	}
	@article{ma2018self,
		title={Self-supervised Sparse-to-Dense: Self-supervised Depth Completion from LiDAR and Monocular Camera},
		author={Ma, Fangchang and Cavalheiro, Guilherme Venturelli and Karaman, Sertac},
		journal={arXiv preprint arXiv:1807.00275},
		year={2018}
	}

Please create a new issue for code-related questions. Pull requests are welcome.
