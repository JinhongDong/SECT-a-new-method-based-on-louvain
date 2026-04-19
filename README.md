# 文章标题

This repository contains the code accompanying the paper **"文章标题."**

文章简介.

接收信息.

---

## 📂 Project Structure

- `norm_dataset/`  
  Contains the input data, including six network files used for training and evaluation.

- `main.py`  
  Main entry point of the project.


- `baselines/`  
  Contains baseline methods used for comparison:
  - `fastgreedy.py`: fastgreedy community detection.
  - `GCN.py`: GCN-based community detection.
  - `lpa.py`: lpa community detection.
  - `louvain.py`: louvain community detection.
  - `leiden.py`: leiden community detection
  - `walktrap.py`: walktrap community detection.

- `utils.py`  
  Utility functions that support the main pipeline (e.g., data loading, preprocessing).

- `result/`  
  Stores the experiment results:
  - `ARI.xlsx`, `NMI.xlsx`, `Q.xlsx`: Summarized evaluation results (Adjusted Rand Index, Normalized Mutual Information, Modularity Q).
  - `拓扑结构.xlsx`: 包含6个数据集的拓扑结构描述
---

## 🚀 Usage

To run the main experiment, use the following command format:

```bash
python main.py --filename <network_name> 
```
**Parameters**:

- `--filename`
Name of the network dataset (e.g., tree, lol, etc.)



**Example**:
```bash
python main.py --filename tree
```

This command uses the tree network.

To reproduce the results of our proposed model and compare it with multiple baselines (CSEA, Node2Vec, LINE, GCN),you can run each baseline individually:
```bash
python baselines/louvain.py      # louvain method
python baselines/lpa.py          # lpa method
python baselines/leiden.py       # leiden method
python baselines/GCN.py          # Graph Convolutional Network
```

## 📈 Result
- ARI.xlsx, NMI.xlsx, and Q.xlsx summarize the evaluation metrics across all datasets.


## 📄 Citation

