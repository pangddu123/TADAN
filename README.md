## Datasets
1. SMD (Server Machine Dataset) is a 5-week-long dataset collected from a large Internet company. You can learn about it
   from [Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network
   ](https://netman.aiops.org/wp-content/uploads/2019/08/OmniAnomaly_camera-ready.pdf).

## 	Get Started

1. Install Python 3.9.13, PyTorch 1.11.0.

2. Train and evaluate. You can reproduce the experiment results as follows:

   ```python
   python main.py --dataset SMD --q 0.005
   ```

## Main Result

The GPU we use is NVIDIA RTX3090 24GB. The following is the F1-score obtained after testing the SMD dataset.

![](./img/result.png)


 
