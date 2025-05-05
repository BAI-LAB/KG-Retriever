# KG-Retriever: Efficient Knowledge Indexing for Retrieval-Augmented Large Language Models

This repository contains the code for the paper "KG-Retriever: Efficient Knowledge Indexing for Retrieval-Augmented Large Language Models"

## Environment Requirements

- Python Version: 3.10.13

## Installation Steps

### STEP 1: Prepare Datasets

1. **Download Datasets and LLMs**:

2. **Preprocess Datasets**:
   - Run src/dataset/preprocess/xxx.py for each datasets
   - Run extract_triples for triples extraction

### STEP 2: Run the Models Demo

1. **Run `run_demo_EX_hotpop_v0.py` to run models in base mode**
2. **Run `run_demo_EX_hotpop_v1.py` to run models in additional modes**

## Citation

If you use this project in your research, please cite the following paper:

```
@article{chen2024kg,
  title={Kg-retriever: Efficient knowledge indexing for retrieval-augmented large language models},
  author={Chen, Weijie and Bai, Ting and Su, Jinbo and Luan, Jian and Liu, Wei and Shi, Chuan},
  journal={arXiv preprint arXiv:2412.05547},
  year={2024}
}
```