import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, seq_len,sort_phase,sort_mode):
        super(VAE, self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(seq_len,128),


            nn.ReLU(),
            nn.Linear(128, 64),

            nn.ReLU(),
            nn.Linear(64,32),

            nn.ReLU(),
            nn.Linear(32,12),


        )
        self.fc_mu = nn.Linear(12, 10)
        self.fc_var = nn.Linear(12, 10)
        self.tanh=nn.Tanh()
        self.decoder=nn.Sequential(
            nn.Linear(10,12),
            nn.ReLU(),
            nn.Linear(12,32),
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32,64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,120),

            nn.ReLU(),
            nn.Linear(120, seq_len),
        )

        # 随机生成隐含向量
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        try:
            x = self.encoder(x)

        except:
            RuntimeError
        mu = self.fc_mu(x)
        var = self.fc_var(x)
        encode=self.reparameterize(mu,var)
        encode=self.tanh(encode)
        decode = self.decoder(encode)
        return encode, decode,mu,var
