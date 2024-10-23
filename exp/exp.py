import os
from time import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import Dataset
from data.preprocess import getData
from model.TADAN import TADAN
from utils.earlystop import EarlyStop
from utils.evaluate import evaluate


class Exp:
    def __init__(self, config):
        self.__dict__.update(config)
        self._get_data()
        self._get_model()

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def _get_data(self):
        data = getData(
            path=self.data_dir,
            dataset=self.dataset,
            period=self.period,
            train_rate=self.train_rate
        )

        self.feature_num = data['train_data'].shape[1]
        self.time_num = data['train_time'].shape[1]
        print('\ndata shape: ')
        for k, v in data.items():
            print(k, ': ', v.shape)

        self.train_set = Dataset(
            data=data['train_data'],
            time=data['train_time'],
            stable=data['train_stable'],
            label=data['train_label'],
            window_size=self.window_size
        )

        self.valid_set = Dataset(
            data=data['valid_data'],
            time=data['valid_time'],
            stable=data['valid_stable'],
            label=data['valid_label'],
            window_size=self.window_size
        )

        self.init_set = Dataset(
            data=data['init_data'],
            time=data['init_time'],
            stable=data['init_stable'],
            label=data['init_label'],
            window_size=self.window_size
        )
        self.test_set = Dataset(
            data=data['test_data'],
            time=data['test_time'],
            stable=data['test_stable'],
            label=data['test_label'],
            window_size=self.window_size
        )

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.init_loader = DataLoader(self.init_set, batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, drop_last=False)

    def _get_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('\ndevice:', self.device)

        self.model = TADAN(
            time_steps=self.time_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            window_size=self.window_size,
            model_dim=self.model_dim,
            ff_dim=self.ff_dim,
            atten_dim=self.atten_dim,
            feature_num=self.feature_num,
            time_num=self.time_num,
            block_num=self.block_num,
            head_num=self.head_num,
            dropout=self.dropout,
            device=self.device,
            d=self.d,
            t=self.t
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.early_stopping = EarlyStop(patience=self.patience, path=self.model_dir + self.dataset + '_model.pkl')
        self.criterion = nn.MSELoss(reduction='mean')


    def _process_one_batch_student(self, batch_data, batch_time, batch_stable, batch_mu, train):
        batch_data = batch_data.float().to(self.device)
        batch_time = batch_time.float().to(self.device)
        batch_stable = batch_stable.float().to(self.device)
        batch_mu = batch_mu.float().to(self.device)

        if train:
            stable_s, trend_s, recon_s = self.model(batch_data, batch_time, self.p)

            noise_s = torch.randn_like(batch_data).to(self.device)
            trend_s_noisy = trend_s + batch_mu

            pred_noise_s = self.model.diffusion.q_sample(batch_data, trend_s,
                                                         torch.full((batch_data.size(0),), self.model.t).to(
                                                             self.device), noise_s)
            student_loss = self.criterion(pred_noise_s, noise_s + batch_mu)

            if self.model.teacher:
                with torch.no_grad():
                    stable_t, trend_t, recon_t = self.model.teacher(batch_data, batch_time)
                    noise_t = noise_s + batch_mu

                    teacher_layers = [stable_t, trend_t, recon_t]
                    student_layers = [stable_s, trend_s, recon_s]
                    assist_loss = 0
                    for i in range(3):
                        cosine_similarity = nn.functional.cosine_similarity(teacher_layers[i], student_layers[i],
                                                                            dim=-1)
                        assist_loss += torch.mean(1 - cosine_similarity)

                total_loss = self.model.eta * student_loss + (1 - self.model.eta) * assist_loss
            else:
                total_loss = student_loss

            return total_loss

        else:
            stable, trend, recon = self.model(batch_data, batch_time, 0.00)
            return stable, trend, recon


    def _compute_multi_view_anomaly_scores(x0, x_t_teacher, x_s_student, intermediate_teacher, intermediate_student,
                                          alpha1=1.0, alpha2=1.0, alpha3=1.0, upsilon=None):
        E_ts = np.linalg.norm(x_t_teacher - x_s_student) ** 2
        if upsilon is None:
            upsilon = [1.0] * len(intermediate_teacher)
        E_inter = sum(upsilon[j] * np.linalg.norm(intermediate_teacher[j] - intermediate_student[j]) ** 2 for j in
                      range(len(intermediate_teacher)))
        E_stu = np.linalg.norm(x0 - x_s_student) ** 2
        AS = alpha1 * E_ts + alpha2 * E_inter + alpha3 * E_stu
        return AS


    def train(self):
        for e in range(self.epochs):
            start = time()

            self.model.train()
            train_loss = []
            for (batch_data, batch_time, batch_stable, _) in tqdm(self.train_loader):
                self.optimizer.zero_grad()
                loss = self._process_one_batch(batch_data, batch_time, batch_stable, train=True)
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                self.model.eval()
                valid_loss = []
                for (batch_data, batch_time, batch_stable, _) in tqdm(self.valid_loader):
                    loss = self._process_one_batch(batch_data, batch_time, batch_stable, train=True)
                    valid_loss.append(loss.item())

            train_loss, valid_loss = np.average(train_loss), np.average(valid_loss)
            end = time()
            print(f'Epoch: {e} || Train Loss: {train_loss:.6f} Valid Loss: {valid_loss:.6f} || Cost: {end - start:.4f}')

            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                break

        self.model.load_state_dict(torch.load(self.model_dir + self.dataset + '_model.pkl'))

    def test(self):
        self.model.load_state_dict(torch.load(self.model_dir + self.dataset + '_model.pkl'))

        with torch.no_grad():
            self.model.eval()
            init_src, init_rec = [], []
            for (batch_data, batch_time, batch_stable, batch_label) in tqdm(self.init_loader):
                _, _, recon = self._process_one_batch(batch_data, batch_time, batch_stable, train=False)
                init_src.append(batch_data.detach().cpu().numpy()[:, -1, :])
                init_rec.append(recon.detach().cpu().numpy()[:, -1, :])

            test_label, test_src, test_rec = [], [], []
            for (batch_data, batch_time, batch_stable, batch_label) in tqdm(self.test_loader):
                _, _, recon = self._process_one_batch(batch_data, batch_time, batch_stable, train=False)
                test_label.append(batch_label.detach().cpu().numpy()[:, -1, :])
                test_src.append(batch_data.detach().cpu().numpy()[:, -1, :])
                test_rec.append(recon.detach().cpu().numpy()[:, -1, :])

        init_src = np.concatenate(init_src, axis=0)
        init_rec = np.concatenate(init_rec, axis=0)
        init_mse = (init_src - init_rec) ** 2

        test_label = np.concatenate(test_label, axis=0)
        test_src = np.concatenate(test_src, axis=0)
        test_rec = np.concatenate(test_rec, axis=0)
        test_mse = (test_src - test_rec) ** 2

        init_score = np.mean(init_mse, axis=-1, keepdims=True)
        test_score = np.mean(test_mse, axis=-1, keepdims=True)

        res = evaluate(init_score.reshape(-1), test_score.reshape(-1), test_label.reshape(-1), q=self.q)
        print("\n=============== " + self.dataset + " ===============")
        print(f"P: {res['precision']:.4f} || R: {res['recall']:.4f} || F1: {res['f1_score']:.4f}")
        print("=============== " + self.dataset + " ===============\n")


