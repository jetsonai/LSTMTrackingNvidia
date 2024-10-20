import random

import numpy as np
import pandas as pd
import scipy

import torch
from torch.utils.data import Dataset, DataLoader, random_split


class trajectory_dataset(Dataset) :
    def __init__(self, opt, csv_path="dataset/WholeVdata2.csv") :
        # Inheritance
        super(trajectory_dataset, self).__init__()

        # Initialize Variable
        self.opt = opt
        self.csv_path = csv_path

        # store X as a list, each element is a 100*42(len*# attributes) np array [velx; vel_y; x; y; acc; angle]*7
        # store Y as a list, each element is a 100*4(len*# attributes) np array[velx; vel_y; x; y]
        self.frames_x, self.frames_y = [], []

        # Function-Calling
        self.load_data()
        self.normalize_data()

    def __len__(self) :
        return len(self.frames_x)

    def __getitem__(self, index) :
        single_data = self.frames_x[index]
        single_label = self.frames_y[index]

        return (single_data, single_label)

    def load_data(self) :
        data_s = pd.read_csv(self.csv_path)
        for vid in data_s.Vehicle_ID.unique() :
            frame_ori = data_s[data_s.Vehicle_ID == vid]
            frame = frame_ori[["Local_X", "Local_Y", "v_Acc", "Angle",
                               "L_rX", "L_rY", "L_rAcc", "L_angle",
                               "F_rX", "F_rY", "F_rAcc", "F_angle",
                               "LL_rX", "LL_rY", "LL_rAcc", "LL_angle",
                               "LF_rX", "LF_rY", "LF_rAcc", "LF_angle",
                               "RL_rX", "RL_rY", "RL_rAcc", "RL_angle",
                               "RF_rX", "RF_rY", "RF_rAcc", "RF_angle"]]
            frame = np.asarray(frame)
            frame[np.where(frame > 4000)] = 0 # assign all 5000 to 0

            # remove anomalies, which has a discontinuious local x or local y
            dis = frame[1:,:2] - frame[:-1,:2]
            dis = np.sqrt(np.power(dis[:,0],2)+np.power(dis[:,1],2))

            index = np.where(dis > 10)
            if not (index[0].all) :
                continue

            # smooth the data column wise
            # window size = 5, polynomial order = 3
            frame =  scipy.signal.savgol_filter(frame, window_length=5, polyorder=3, axis=0)

            # calculate vel_x and vel_y according to localX and localY for all vehicles
            all_veh = []

            for i in range(7) :
                vel_x = (frame[1:,0+i*4]-frame[:-1, 0+i*4])/0.1
                vel_x_avg = (vel_x[1:]+vel_x[:-1])/2.0
                vel_x1 = [2.0*vel_x[0]- vel_x_avg[0]]
                vel_x_end = [2.0*vel_x[-1]- vel_x_avg[-1]];
                vel_x = np.array(vel_x1 + vel_x_avg.tolist() + vel_x_end)

                vel_y = (frame[1:,1+i*4]-frame[:-1, 1+i*4])/0.1
                vel_y_avg = (vel_y[1:]+vel_y[:-1])/2.0
                vel_y1 = [2.0*vel_y[0]- vel_y_avg[0]]
                vel_y_end = [2.0*vel_y[-1]-vel_y_avg[-1]]
                vel_y = np.array(vel_y1 + vel_y_avg.tolist() + vel_y_end)

                if isinstance(all_veh,(list)) :
                    all_veh = np.vstack((vel_x, vel_y))
                else:
                    all_veh = np.vstack((all_veh, vel_x.reshape(1,-1)))
                    all_veh = np.vstack((all_veh, vel_y.reshape(1,-1)))

            all_veh = np.transpose(all_veh)
            total_frame_data = np.concatenate((all_veh[:,:2], frame), axis=1)

            # split into several frames each frame have a total length of 100, drop sequence smaller than 130
            if total_frame_data.shape[0] < 130 :
                continue
            
            # Slice Input & Target Variable
            X = total_frame_data[:-29,:]
            Y = total_frame_data[29:,:4]

            count = 0
            for i in range(X.shape[0]-100) :
                if random.random() > 0.2 :
                    continue

                if count > 20 :
                    break

                self.frames_x = self.frames_x + [X[i:i+100,:]]
                self.frames_y = self.frames_y + [Y[i:i+100,:]]

                count += 1

    def normalize_data(self) :
        # Compute Stats & Normalize Dataset
        A = [list(x) for x in zip(*(self.frames_x))]
        A = torch.tensor(np.array(A), dtype=torch.float32)
        A = A.view(-1, A.shape[2])
        
        self.mn = torch.mean(A, dim=0)
        self.range = (torch.max(A, dim=0).values - torch.min(A, dim=0).values)/2.0
        self.range = torch.ones(self.range.shape, dtype=torch.float32)
        self.std = torch.std(A,dim=0)
        self.frames_x = [(torch.tensor(item, dtype=torch.float32)-self.mn)/(self.std*self.range) for item in self.frames_x]
        self.frames_y = [(torch.tensor(item, dtype=torch.float32)-self.mn[:4])/(self.std[:4]*self.range[:4]) for item in self.frames_y]


def get_dataloader(opt, csv_path="dataset/WholeVdata2.csv") :
    # Load Dataset
    dataset = trajectory_dataset(opt, csv_path)

    # Split Dataset
    num_train = int(dataset.__len__()*opt.train_ratio)
    num_test = int(dataset.__len__()*opt.test_ratio) - num_train
    num_valid = int(dataset.__len__() - num_test - num_train)
    train, valid, test = random_split(dataset, [num_train, num_valid, num_test])

    # Create Dataloader Instance
    train_loader = DataLoader(train, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(valid, batch_size=opt.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test, batch_size=opt.batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, dataset