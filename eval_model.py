import torch

def eval_VAE(model,data_loader):

    loss_1 = 0

    kld_loss=0

    loss_F = torch.nn.MSELoss()

    with torch.no_grad():
        for step, (H_V, K_V, phase_point,phase_label,M_label,T_length) in enumerate(data_loader):



            H_V=H_V.float().cuda()
            K_V=K_V.float().cuda()
            phase_point=phase_point.float().cuda()

            input_x = torch.cat((H_V,K_V),1)
            input_x=torch.cat((input_x,phase_point),1)
            M_label=M_label.squeeze_().long().cuda()

            encoded, pred, mu,log_var = model(input_x)
            loss_1 += loss_F(input_x, pred)
            kld_loss += torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)



    loss_1 /= len(data_loader)
    kld_loss /=len(data_loader)
    # print("Avg src Loss = {}".format(loss))
    return loss_1,kld_loss