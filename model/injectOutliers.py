import numpy as np
import os


# 加载数据
def load_data(data_dir):
    train_data = np.load(os.path.join(data_dir, 'SMD_train_data.npy'))
    test_data = np.load(os.path.join(data_dir, 'SMD_test_data.npy'))
    test_label = np.load(os.path.join(data_dir, 'SMD_test_label.npy'))
    return train_data, test_data, test_label


# 注入全局异常
def inject_global_outliers(data, gamma=3):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    num_outliers = int(0.05 * data.shape[0])  # 5%的全局异常
    indices = np.random.choice(data.shape[0], num_outliers, replace=False)

    data[indices] = mu + gamma * sigma
    return data


# 注入上下文异常
def inject_contextual_outliers(data, window_size=5, gamma=3):
    for i in range(window_size, len(data) - window_size):
        mu = np.mean(data[i - window_size:i + window_size], axis=0)
        sigma = np.std(data[i - window_size:i + window_size], axis=0)
        if np.random.rand() < 0.05:  # 5%的上下文异常
            data[i] = mu + gamma * sigma
    return data


# 注入季节性异常
def inject_seasonal_outliers(data, omega=0.5, gamma=0.2):
    t = np.arange(len(data))
    seasonal_component = np.sin(2 * np.pi * omega * t) * gamma
    data += seasonal_component[:, None]
    return data


# 注入趋势异常
def inject_trend_outliers(data, gamma=0.01):
    t = np.arange(len(data))
    trend_component = gamma * t
    data += trend_component[:, None]
    return data


# 保存数据
def save_data(data_dir, modified_data, file_name):
    np.save(os.path.join(data_dir, file_name), modified_data)


def main():
    data_dir = '/path/to/SMD/'  # 修改为SMD文件夹的路径
    train_data, test_data, test_label = load_data(data_dir)

    # 注入异常
    train_data_with_global_outliers = inject_global_outliers(train_data.copy())
    test_data_with_contextual_outliers = inject_contextual_outliers(test_data.copy())
    test_data_with_seasonal_outliers = inject_seasonal_outliers(test_data.copy())
    test_data_with_trend_outliers = inject_trend_outliers(test_data.copy())

    # 保存注入异常的数据
    save_data(data_dir, train_data_with_global_outliers, 'SMD_train_data_with_global_outliers.npy')
    save_data(data_dir, test_data_with_contextual_outliers, 'SMD_test_data_with_contextual_outliers.npy')
    save_data(data_dir, test_data_with_seasonal_outliers, 'SMD_test_data_with_seasonal_outliers.npy')
    save_data(data_dir, test_data_with_trend_outliers, 'SMD_test_data_with_trend_outliers.npy')


if __name__ == '__main__':
    main()
