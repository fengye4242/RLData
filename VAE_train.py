import torch.optim as optim
from  get_data  import My_120_norm_Dataset
from  utils import get_data_loader
from utils import save_model
from eval_model  import eval_VAE
import params
from autoencoder import VAE

import torch
from torch import nn
import torch.nn.functional as F
if __name__ == '__main__':
    class MyModule(nn.Module):
        def __init__(self, num):
            super(MyModule, self).__init__()
            params = torch.zeros(num, requires_grad=True)
            self.params = nn.Parameter(params)

        def forward(self, x):
            y = self.params * x

            return y


    def kl_divergence(p, q):
        '''
        args:
            2 tensors `p` and `q`
        returns:
            kl divergence between the softmax of `p` and `q`
        '''

        s1 = torch.sum(p * torch.log(p / q), 1)
        s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)), 1)
        return torch.mean(s1 + s2)

    train_data_root = 'example/vae_train_normed_data.mat'
    test_data_root = 'examlple/vae_test_normed_data.mat'


    train_data = My_120_norm_Dataset(train_data_root)
    test_data = My_120_norm_Dataset(test_data_root)

    train_loader = get_data_loader(train_data, train_set=False, train_batch_size=params.batch_size,
                                   shuffle=True)
    test_loader = get_data_loader(test_data, train_set=False, train_batch_size=params.batch_size,
                                  shuffle=True)


    log_var_model=MyModule(10)
    log_var_model.cuda()
    loss_F = torch.nn.MSELoss()
    loss_c = torch.nn.CrossEntropyLoss()

    model = VAE(243, 3, 5)
    model.cuda()

    optimizer = optim.Adam([{"params":model.parameters()},
                           {"params": log_var_model.parameters()}])

    beta=0.01
    current_save_name = "vae_auto_encoder_beta_"+str(int(beta*1000))+"_train"
    for epoch in range(10000):
        for step, (H_V,K_V,phase_point,phase_label,M_label,T_length) in enumerate(train_loader):
            H_V = H_V.float().cuda()
            K_V = K_V.float().cuda()
            phase_point=phase_point.float().cuda()
            input_x = torch.cat((H_V, K_V), 1)
            input_x = torch.cat((input_x, phase_point), 1)
            M_label = M_label.squeeze_().long().cuda()
            encoded, decoded, mu, log_var = model(input_x)
            loss_1 = F.mse_loss(decoded, input_x, reduction="mean")
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

            loss = loss_1 + beta * kld_loss.abs()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss, train_kld_loss = eval_VAE(model, train_loader)
        test_loss, test_kld_loss = eval_VAE(model, test_loader)

        if epoch % 10 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy())
        if epoch % 1000 == 0:
            save_name = current_save_name + "-{}.pt".format(epoch + 1)
            save_model(model, save_name)
    save_name = current_save_name + "-{}.pt".format(epoch + 1)
    save_model(model, save_name)