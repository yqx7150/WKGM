# SVD-WKGM
**Paper**: WKGM: Weight-K-space Generative Model for Parallel Imaging Reconstruction

**Authors**: Zongjiang Tu, Die Liu, Xiaoqing Wang, Chen Jiang, Minghui Zhang, Shanshan Wang, Qiegen Liu*, Dong Liang*

Date : June-14-2022  
Version : 1.0  
The code and the algorithm are for non-comercial use only.  
Copyright 2022, Department of Electronic Information Engineering, Nanchang University. 

Deep learning based parallel Imaging (PI) has made great progresses in recent years to accelerate magnetic resonance imaging (MRI). Nevertheless, the performanc-es and robustness of existing methods can still be im-proved. In this work, we propose to explore the k-space domain learning via robust generative modeling for flexible PI reconstruction, coined weight-k-space genera-tive model (WKGM). Specifically, WKGM is a general-ized k-space domain model, where the k-space weighting technology and high-dimensional space augmentation design are efficiently incorporated for score-based gen-erative model training, resulting in good and robust re-constructions. In addition, WKGM is flexible and thus can be synergistically combined with various traditional k-space PI models, generating learning-based priors to produce high-fidelity reconstructions. Experimental re-sults on datasets with varying sampling patterns and ac-celeration factors demonstrate that WKGM can attain state-of-the-art reconstruction results with the well-learned k-space generative prior.

## Training Demo
```bash
python main.py --config=configs/ve/SIAT_kdata_ncsnpp.py --workdir=exp --mode=train --eval_folder=result
```
## Test Demo
```bash
python demo_svdWKGM.py
```
## Checkpoints
We provide pretrained checkpoints. You can download pretrained models from [Google Drive] (https://drive.google.com/file/d/1DmRTPmc_xYaVO3pX1R_CE0ZpiBRFkCwG/view?usp=sharing)
## Graphical representation
## Pipeline of the prior learning process and PI reconstruction procedure in WKGM
<div align="center"><img src="https://github.com/yqx7150/SVD-WKGM/blob/main/Fig-1.png" >  </div>
Top line: Prior learning is conducted in weight-k-space domain at a single coil. Bottom line: PI reconstruction is conducted in iterative scheme that alternates between WKGM update and other traditional iterative methods.

## Illustration of the forward and reverse processes of k-space data.
<div align="center"><img src="https://github.com/yqx7150/SVD-WKGM/blob/main/Fig-2.png" >  </div>


