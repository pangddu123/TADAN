## Datasets
1. PSM (PooledServer Metrics) is collected internally from multiple application server nodes at eBay.
   You can learn about it
   from [Practical Approach to Asynchronous Multivariate Time Series Anomaly Detection and Localization](https://dl.acm.org/doi/abs/10.1145/3447548.3467174)
   .
2. MSL (Mars Science Laboratory rover) is a public dataset from NASA. You can learn about it 
   from [Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding](https://arxiv.org/pdf/1802.04431.pdf).
3. MIT-BIH (Arrhythmia Dataset) comprises 48 half-hour two-lead electrocardiogram (ECG) recordings, encompassing a diverse range of heartbeat. You can learn about it
   from [PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals](https://www.ahajournals.org/doi/epub/10.1161/01.CIR.101.23.e215).
4. SMD (Server Machine Dataset) is a 5-week-long dataset collected from a large Internet company. You can learn about it
   from [Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network
   ](https://netman.aiops.org/wp-content/uploads/2019/08/OmniAnomaly_camera-ready.pdf).
5. SWaT (Secure Water Treatment) is obtained from 51 sensors of the critical infrastructure system under continuous
   operations. You can learn about it from [SWaT: a water treatment testbed for research and training on ICS security
   ](https://ieeexplore.ieee.org/abstract/document/7469060).

## 	Get Started

1. Install Python 3.9.13, PyTorch 1.11.0.

2. Train and evaluate. You can reproduce the experiment results as follows:

   ```python
   python main.py --dataset SMD --q 0.005
   ```

## Main Result

The GPU we use is NVIDIA RTX3090 24GB. The following is the F1-score obtained after testing the SMD dataset.

![](./img/result.png)


 
