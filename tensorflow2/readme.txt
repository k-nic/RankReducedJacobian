# Reproducible Implementation of Low-Rank Jacobian Regularization

This repository contains the TensorFlow 2 implementation of the alternating training algorithm for autoencoders with low-rank Jacobian regularization, as proposed and analyzed in the paper:

Rustem Takhanov, Y. Sultan Abylkairov, Maxat Tezekbayev, "Autoencoders for a manifold learning problem with a jacobian rank constraint", Pattern Recognition, Volume 143, 2023, 109777.
https://doi.org/10.1016/j.patcog.2023.109777

---

## Citation

If you use this code in your research or publications, please cite the following paper:

```bibtex
@article{takhanov2024learning,
  title={Autoencoders for a manifold learning problem with a jacobian rank constraint},
  author={Rustem Takhanov, Y. Sultan Abylkairov, Maxat Tezekbayev},
  journal={Pattern Recognition},
  volume={143},
  year={2023},
  publisher={Elsevier}
}
```

---

## Requirements

* Python ≥ 3.10
* TensorFlow ≥ 2.12
* NumPy ≥ 1.24

## Usage

Example training notebook for MNIST dataset is provided.

## License

This code is released for research and educational purposes only.
Please cite the above paper if you use or modify this repository in your work.
