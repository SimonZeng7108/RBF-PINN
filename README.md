# RBF-PINN
This is the official repo for the paper **[Training Dynamics in Physics-Informed Neural Networks with Feature Mapping](https://arxiv.org/abs/2402.06955)**.<br/>
**Preprint, Under Review**<br/>
Chengxi Zeng, Tilo Burghardtg, Alberto M Gambaruto<br/>

Short paper: **[RBF-PINN: Non-Fourier Positional Embedding in Physics-Informed Neural Networks](https://arxiv.org/abs/2402.08367)**.<br/>
International Conference on Learning Representations (ICLR 2024), AI4DifferentialEquations in Science Workshop

<img src="https://github.com/SimonZeng7108/RBF-PINN/blob/master/Figs/lorenz.gif" width="800" height="500"><br/>

## Abstract
Physics-Informed Neural Networks (PINNs) have emerged as an iconic machine learning approach for solving Partial Differential Equations (PDEs). Although its variants have achieved significant progress, the empirical success of utilising feature mapping from the wider Implicit Neural Representations studies has been substantially neglected. We investigate the training dynamics of PINNs with a feature mapping layer via the limiting Conjugate Kernel and Neural Tangent Kernel, which sheds light on the convergence and generalisation of the model. We also show the inadequacy of commonly used Fourier-based feature mapping in some scenarios and propose the conditional positive definite Radial Basis Function as a better alternative. The empirical results reveal the efficacy of our method in diverse forward and inverse problem sets. This simple technique can be easily implemented in coordinate input networks and benefits the broad PINNs research.

## Repo usage
### Requirements 
```Bash
conda create -n RBF-PINN python=3.8
conda activate RBF-PINN
pip install [following...]
```
- `torch == 2.0`
- `torchsummary`
- `numpy == 1.24.3`
- `matplotlib`
- `pyDOE`

## Citation
```
@misc{zeng2024training,
      title={Training dynamics in Physics-Informed Neural Networks with feature mapping}, 
      author={Chengxi Zeng and Tilo Burghardt and Alberto M Gambaruto},
      year={2024},
      eprint={2402.06955},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

