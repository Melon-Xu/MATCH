# MATCH: Multi-faceted Adaptive Topo-Consistency for Semi-Supervised Histopathology Segmentation

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://arxiv.org/abs/2510.01532)
[![arXiv](https://img.shields.io/badge/arXiv-2510.01532-b31b1b.svg)](https://arxiv.org/abs/2510.01532)

This is the official implementation of our paper "MATCH: Multi-faceted Adaptive Topo-Consistency for Semi-Supervised Histopathology Segmentation", which has been accepted by **NeurIPS 2025**.


## 📂 Code Structure

This repository contains the following core modules:

- `match_pair.py`: Implements **MATCH-Pair**, which facilitates accurate matching between two persistence diagrams. It ensures precise correspondence of topological features between pairs of predictions.
- `match_global.py`: Implements **MATCH-Global**, designed for accurate matching among multiple persistence diagrams. It aligns topological structures globally across multiple predictions to ensure coherence.
- `dual_level_topo_consistency.py`: Implements the **Dual-Level Topological Consistency Loss**. This loss function enforces topological consistency and can be seamlessly incorporated into any semi-supervised segmentation framework to improve structural accuracy.


## 📊 Experimental Results

Our method has been evaluated on multiple histopathology image datasets, effectively reducing topological errors and improving segmentation accuracy. Please refer to our paper for detailed results.



## 📝 Citation

If you find our work helpful for your research, please consider citing:

```bibtex
@inproceedings{xu2025match,
  title={MATCH: Multi-faceted Adaptive Topo-Consistency for Semi-Supervised Histopathology Segmentation},
  author={Xu, Meilong and Hu, Xiaoling and Abousamra, Shahira and Li, Chen and Chen, Chao},
  booktitle={NeurIPS},
  year={2025}
}
```

## 📧 Contact

If you have any questions, please contact us via issues or email.

Email: meixu@cs.stonybrook.edu
