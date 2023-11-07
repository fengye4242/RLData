import gym
from gym import spaces
import numpy as np
import params
import torch
import sys
from scipy.interpolate import interp1d
sys.path.append('./')

stack_num=2 #define the number of history observation

def rolling_window(data, length, window, step):
# split data using sliding Windows
    data_frame = np.int((length / step))+1
    out = np.zeros((data_frame, window))
    for i in range(data_frame):
            out[i] = data[i * step:i * step + window]
    return out

class Trajectory(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self,data_loader,init_data,init_obs_dataset,model):
        super(Trajectory,self).__init__()

        model.cpu()
        self.data_loader = data_loader
        self.iter_data=self.data_loader.__iter__()
        self.encoder=model.encoder
        self.decoder=model.decoder
        self.init_data=init_data
        self.init_obs_dataset=init_obs_dataset
        self.label_index_range=np.zeros((5,2))
        self.mu = model.fc_mu
        self.tanh = model.tanh

        for i in range(5):
            i_range=np.argwhere(init_data.M_label==i)[:,0]
            self.label_index_range[i,0]=i_range.max()
            self.label_index_range[i,1]=i_range.min()



        #观测空间
        self.encoder_high =[1 for i in range(10)]
        self.encoder_low=[-1 for i in range(10)]

        self.cycle_length_high=[1]
        self.cycle_length_low = [0]


        current_position_high=[1]
        current_position_low = [0]



        delta_z_high = [2 for i in range(60)]
        delta_z_low=[-2 for i in range(60)]


        #动作空间
        action_low=self.encoder_high
        action_high=self.encoder_low

        phase_label_hish = [5]
        phase_label_low = [0]
        observation_low=self.encoder_low+self.cycle_length_low+current_position_low+delta_z_low+phase_label_low
        observation_high=self.encoder_high+self.cycle_length_high+current_position_high+delta_z_high+phase_label_hish
        self.action_space=spaces.Box(low=np.array(action_low),high=np.array(action_high))
        self.observation_space=spaces.Box(low=np.array(observation_low),high=np.array(observation_high))
        self.stack_observation=np.zeros((params.stack_num,len(observation_low)))



    def reset(self):

        try:

            H_V,K_V,phase_point,phase_label,M_label,Traj_length = next(self.iter_data)

        except StopIteration:
            self.iter_data = self.data_loader.__iter__()
            H_V,K_V,phase_point,phase_label,M_label,Traj_length = next(self.iter_data) #M_label:locomotion mode
        init_obs_label = M_label[0][0]
        init_label_range=self.label_index_range[init_obs_label]
        choice_num_init = np.random.randint(init_label_range[1], init_label_range[0])


        #data used for initialize  observation
        c_init_traj_H=self.init_data.H_V[choice_num_init]#used for latent variable initialization
        c_init_traj_K=self.init_data.K_V[choice_num_init]
        c_init_obs_H=self.init_obs_dataset.H_V[choice_num_init]
        c_init_obs_K=self.init_obs_dataset.K_V[choice_num_init]
        c_init_original_length=self.init_obs_dataset.T_length[choice_num_init]
        c_init_obs_phase_label=self.init_obs_dataset.phase_label[choice_num_init]
        c_init_obs_phase_label=c_init_obs_phase_label[:c_init_original_length[0]]
        c_init_obs_phase_point=self.init_obs_dataset.phase_point[choice_num_init]

        c_init_segment_length = np.random.choice(
            int(params.window_length / 2)) + stack_num * params.step

        c_init_obs_H=c_init_obs_H[:c_init_original_length[0]]
        c_init_obs_K=c_init_obs_K[:c_init_original_length[0]]

        c_init_obs_H_V_segment=torch.from_numpy(c_init_obs_H[-c_init_segment_length:])
        c_init_obs_K_V_segment=torch.from_numpy(c_init_obs_K[-c_init_segment_length:])
        c_init_obs_phase_label_segment = c_init_obs_phase_label[-c_init_segment_length:]

        #The environment contains two gait trajectories
        c_first_traj_H=H_V[0][0]
        c_first_traj_K=K_V[0][0]
        c_first_traj_length=Traj_length[0][0].item()
        c_first_true_traj_H=c_first_traj_H[:c_first_traj_length]
        c_first_true_traj_K = c_first_traj_K[:c_first_traj_length]

        H_V=H_V[0][1]
        K_V = K_V[0][1]
        Traj_length=Traj_length[0][1]

        self.H_V=H_V
        self.K_V=K_V
        self.Traj_length=Traj_length


        con_H=torch.cat([c_init_obs_H_V_segment,c_first_true_traj_H])
        con_K=torch.cat([c_init_obs_K_V_segment,c_first_true_traj_K])

        con_H=torch.cat([con_H,H_V[:Traj_length]])
        con_K=torch.cat([con_K,K_V[:Traj_length]])

        ##Extended trajectories are used for prediction accuracy comparisons
        con_H=torch.cat([con_H,H_V[:Traj_length]])
        con_K=torch.cat([con_K,K_V[:Traj_length]])
        ##
        con_length=c_init_segment_length+torch.tensor(c_first_traj_length).float()+Traj_length.float()
        input_H = rolling_window(con_H,con_length, params.window_length, params.step)
        input_K = rolling_window(con_K,con_length,params.window_length,params.step)

        #compute the trajectory length after window splitting
        for i in range(len(input_H)):
            if i == 0:
                c_K = input_K[i][0:15]
                c_H = input_H[i][0:15]

            else:
                c_K = np.append(c_K, input_K[i][0:15])
                c_H = np.append(c_H, input_H[i][0:15])
        c_K=np.append(c_K,input_K[i][15:30])
        c_H =np.append(c_H,input_H[i][15:30])

        extend_length=len(c_K)-c_init_segment_length-c_first_traj_length-Traj_length.item()

        #caculate the gait phase and trajectory length of current phase
        c_init_index_1=np.array([i for i in range(c_init_original_length[0])])/(c_init_original_length[0]-1)
        c_init_index=c_init_index_1[-c_init_segment_length:]
        c_first_index=np.array([i for i in range(c_first_traj_length)])/(c_first_traj_length-1)
        c_L_index=np.array([i for i in range(Traj_length.item())])/(Traj_length.item()-1)
        c_exted=np.array([i for i in range(extend_length)])/(Traj_length.item()-1)


        c_init_length_index=[c_init_original_length[0] for i in range(c_init_segment_length)]
        c_first_length_index=[c_first_traj_length for i in range(c_first_traj_length)]
        c_L_length_index=[Traj_length.item() for i in range(Traj_length.item())]

        c_exted_length_index=[Traj_length.item() for i in range(extend_length)]
        self.con_index=np.hstack([c_init_index,c_first_index,c_L_index,c_exted])
        self.true_length=np.hstack([c_init_length_index,c_first_length_index,c_L_length_index,c_exted_length_index])



        self.input_H = input_H
        self.input_K = input_K


        c_first_true_traj_phase_label=phase_label[0][0][:c_first_traj_length]
        phase_label=phase_label[0][1]
        con_phase_label=torch.cat([torch.from_numpy(c_init_obs_phase_label_segment),c_first_true_traj_phase_label])
        con_phase_label = torch.cat([con_phase_label, phase_label[:Traj_length]])
      #  con_phase_label=torch.cat([con_phase_label, phase_label[ :Traj_length]])
        ##
        con_phase_label=torch.cat([con_phase_label, phase_label[:Traj_length]])
        con_phase_label=torch.cat([con_phase_label, phase_label[:Traj_length]])
        ##
        input_phase_label=rolling_window(con_phase_label,con_length,params.window_length,params.step)
        self.input_phase_label=input_phase_label.astype('int64')

        #caculate the estimated trajectory length
        c_length_label=np.zeros((len(con_phase_label)))

        c_length_label[:c_init_segment_length+c_first_traj_length]=self.true_length[:c_init_segment_length+c_first_traj_length]
        find_diff_phase=np.where((np.append(con_phase_label[1:],con_phase_label[-1])-con_phase_label.numpy())!=0)[0]+1
        first_appear_index=find_diff_phase
        first_appear_phase_label=con_phase_label[first_appear_index]
        remove_init=np.where(first_appear_phase_label==0)[0]
        first_appear_index=first_appear_index[remove_init[0]:]
        first_appear_phase_label=first_appear_phase_label[remove_init[0]:]
        for i in range(4):
            c_phase=np.where(first_appear_phase_label==i)[0]
            count_c=len(c_phase)
            for ii in range(count_c-1):
                c_length=first_appear_index[c_phase[ii+1]]-first_appear_index[c_phase[ii]]
                start_index=first_appear_index[c_phase[ii+1]]
                if i == first_appear_phase_label[-1] and ii == count_c-2:
                    end_index=len(c_length_label)
                else:
                    end_index=first_appear_index[c_phase[ii+1]+1]
                c_length_label[start_index:end_index]=c_length

        input_length_label = c_length_label[:len(c_K)]
        self.input_length_label=input_length_label

        c_init_traj= np.concatenate((c_init_traj_H, c_init_traj_K))
        c_init_traj=np.concatenate((c_init_traj,c_init_obs_phase_point))
        self.encoder_val=self.tanh(self.mu(self.encoder(torch.Tensor(c_init_traj)))).detach().numpy()

        self.offset=0
        self.cycle_length=[0.5]

        self.window_ratio=params.window_length/self.cycle_length[0]
        self.step_ratio=params.step/self.cycle_length[0]
        self.current_position=0

        self.index_ratio_start=self.current_position
        self.index_ratio_end=self.index_ratio_start+self.window_ratio

        self.c_step = -1
        self.true_index=False

        #initialize the gait phase
        self.index_ratio_start = self.con_index[self.c_step * params.step]
        self.index_ratio_end = self.con_index[self.c_step * params.step + params.window_length]

        self.init_step=True
        for init_step_count in range(params.stack_num):
            action=np.zeros(self.action_space.shape)
            c_obs=self.step(action)
            self.stack_observation[init_step_count,:]=c_obs
        self.init_step=False
        return self.stack_observation


    def _next_observation(self):
        torch_encoder=torch.Tensor(self.encoder_val)
        torch_trajectory=self.decoder((torch_encoder))
        torch_trajectory_H=torch_trajectory[0:120]
        torch_trajectory_K=torch_trajectory[120:240]
        torch_phase_z = torch_trajectory[240:243]
        c_trajectory_H=torch_trajectory_H.cpu().detach().numpy().tolist()
        c_trajectory_K=torch_trajectory_K.cpu().detach().numpy().tolist()

        numpy_trajectory_phase_point = torch_phase_z.cpu().detach().numpy().tolist()
        c_trajectory_phase_label = np.zeros(len(torch_trajectory_H))
        for phase_num in range(len(numpy_trajectory_phase_point)):
            c_location = round(numpy_trajectory_phase_point[phase_num] * len(torch_trajectory_H))
            if phase_num == 0:
                c_trajectory_phase_label[0:c_location] = 0
            else:
                c_trajectory_phase_label[
                round(numpy_trajectory_phase_point[phase_num - 1] * len(torch_trajectory_H)):c_location] = phase_num
        c_trajectory_phase_label[c_location:] = 3

        index_ratio_start=self.index_ratio_start
        index_ratio_end=self.index_ratio_end

        phase_error = 0
        c_prediction_location=index_ratio_start*self.cycle_length[0]
        numpy_trajectory_phase_point=torch_phase_z.cpu().detach().numpy()
        include_HS_phase=np.append(numpy_trajectory_phase_point,1)
        predicted_phase = [0, 1, 2, 3]
        current_phase=self.input_phase_label[self.c_step]
        b =np.hstack((current_phase[1:], current_phase[-1]))
        phase_point_index = np.argwhere((current_phase - b) != 0) + 1
        if len(phase_point_index)>0:
            true_label = current_phase[phase_point_index]

            predicted_phase_location = include_HS_phase*self.cycle_length[0]


            for i in range(len(phase_point_index)):
                relative_time = predicted_phase_location - c_prediction_location - np.squeeze(phase_point_index[i])
                predicted_phase_index=np.argwhere((predicted_phase - true_label[i])==0)
                corresponding_predict_phase_relative_time=relative_time[np.squeeze(predicted_phase_index)]
                if np.abs(corresponding_predict_phase_relative_time)>(self.cycle_length[0]/2):
                    if corresponding_predict_phase_relative_time<=0:
                        corresponding_predict_phase_relative_time=corresponding_predict_phase_relative_time+self.cycle_length[0]
                    else:
                        corresponding_predict_phase_relative_time=corresponding_predict_phase_relative_time-self.cycle_length[0]
                phase_error=np.abs(corresponding_predict_phase_relative_time)+phase_error

        if index_ratio_start<index_ratio_end and index_ratio_end<1:
            xx = np.linspace(0, 119, 120)
            norm_H = interp1d(xx,c_trajectory_H,kind='cubic')
            norm_K = interp1d(xx,c_trajectory_K,kind='cubic')
            interpolation_index=np.linspace((params.stander_cycle_length-1)*index_ratio_start,(params.stander_cycle_length-1)*index_ratio_end,params.window_length)#linspace(0,120)120个点，数据段跨度长度为119

            self.interpolation_point_H=norm_H(interpolation_index)
            self.interpolation_point_K=norm_K(interpolation_index)
            self.interpolation_point_phase_label = np.interp(interpolation_index,
                                                             [i for i in range(params.stander_cycle_length)],
                                                             c_trajectory_phase_label)
           # self.interpolation_point_H=np.interp(interpolation_index,[i for i in range(params.stander_cycle_length)],c_trajectory_H)
            #self.interpolation_point_K = np.interp(interpolation_index,[i for i in range(params.stander_cycle_length)], c_trajectory_K)

        else:
            if index_ratio_end>=1:
                index_ratio_end=index_ratio_end-1
            xx = np.linspace(0, 120*2-1, 120*2)
            norm_H = interp1d(xx,np.append(c_trajectory_H,c_trajectory_H),kind='cubic')
            norm_K = interp1d(xx,np.append(c_trajectory_K,c_trajectory_K),kind='cubic')
            interpolation_index=np.linspace((params.stander_cycle_length-1)*index_ratio_start,(params.stander_cycle_length-1)*(index_ratio_end+1),params.window_length)
            self.interpolation_point_H=norm_H(interpolation_index)
            self.interpolation_point_K=norm_K(interpolation_index)
            self.interpolation_point_phase_label = np.interp(interpolation_index,
                                                             [i for i in range(params.stander_cycle_length)],
                                                             c_trajectory_phase_label)
           # self.interpolation_point_H=np.interp(interpolation_index,[i for i in range(2*params.stander_cycle_length)],np.append(c_trajectory_H,c_trajectory_H))
           # self.interpolation_point_K=np.interp(interpolation_index,[i for i in range(2*params.stander_cycle_length)],np.append(c_trajectory_K,c_trajectory_K))
        delta_1=self.input_H[self.c_step]-self.interpolation_point_H
        delta_2=self.input_K[self.c_step]-self.interpolation_point_K
        delta_z=np.concatenate((delta_1,delta_2))


        if np.abs(delta_z).any()>2:
            print('need')


        obs=np.concatenate((np.array(self.encoder_val),np.array(self.cycle_length),np.array([self.current_position]),delta_z,[phase_error]))

        return obs

    def _take_action(self,action):
        self.encoder_val=np.clip(self.encoder_val+action[0:10],self.encoder_low,self.encoder_high)


        # estimated length
        self.cycle_length = np.array([self.input_length_label[(self.c_step + 1) * params.step]])



    def step(self, action):

            self._take_action(action)
            done=False

            #窗口移动要和c_step移动一致，现在得到的是下一个观测
            # c_position=self.current_position[0]+self.step_ratio

            self.window_ratio = params.window_length / self.cycle_length[0]
            self.step_ratio = params.step / self.cycle_length[0]

            c_position = self.current_position + self.step_ratio+self.offset
            # if c_position<=-1 or c_position>=2:
            #     print(c_position)
            if c_position<0:
                c_position=c_position-int(c_position)+1
            elif c_position>=1:
                c_position=c_position-int(c_position)
            self.current_position=c_position


            self.index_ratio_start=self.current_position
            self.index_ratio_end=self.index_ratio_start+self.window_ratio

            #c_step移动
            self.c_step+=1

            ###在这里测试使用真实的起始点与结束点
            if self.con_index[self.c_step * params.step] > self.con_index[self.c_step * params.step + params.window_length - 1]:
                self.index_ratio_start = self.con_index[self.c_step * params.step]
                self.index_ratio_end = self.con_index[self.c_step * params.step + params.window_length - 1]

                self.current_position = self.index_ratio_start

            obs = self._next_observation()



            if self.init_step == False:
                self.stack_observation[:-1, :] = self.stack_observation[1:, :]
                self.stack_observation[-1, :] = obs

            reward = -np.sum(np.abs(obs[12:72]))-0.1*obs[72]-np.sum(np.abs(action))



            if self.c_step>=len(self.input_H)-1:
                done=True

                c_encoder = torch.Tensor(self.encoder_val)
                predicted_norm_trajectory = self.decoder(c_encoder)
                pred_traj = predicted_norm_trajectory.cpu().detach().numpy()


                xx = np.linspace(0, self.Traj_length-1,  self.Traj_length)
                true_H_V = interp1d(xx, self.H_V[:self.Traj_length], kind='cubic')
                true_K_V = interp1d(xx, self.K_V[:self.Traj_length], kind='cubic')

                interpolation_index = np.linspace(0,self.Traj_length-1,params.stander_cycle_length)
                norm_true_H_V=true_H_V(interpolation_index)
                norm_true_K_V=true_K_V(interpolation_index)
                final_error=pred_traj[:240]-np.append(norm_true_H_V,norm_true_K_V)

                reward= -np.sum(np.abs(obs[12:72]))-0.1*obs[72]-np.sum(np.abs(final_error))-np.sum(np.abs(action))
            if self.init_step == True:
                return obs
            else:
                return self.stack_observation,reward,done,{}

