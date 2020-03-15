import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryNetworkDQN(nn.Module):
    def __init__(self, indexes_full_state=10 * 128, input_size=38, input_size_subset=38, sim_size=64):
        super(QueryNetworkDQN, self).__init__()
        self.conv1_s = nn.Conv1d(input_size_subset, 256, 1)
        self.bn1_s = nn.BatchNorm1d(input_size_subset)
        self.conv2_s = nn.Conv1d(256, 128, 1)
        self.bn2_s = nn.BatchNorm1d(256)
        self.conv3_s = nn.Conv1d(128, 1, 1)
        self.bn3_s = nn.BatchNorm1d(128)

        self.linear_s = nn.Linear(indexes_full_state, 128)
        self.bn_last_s = nn.BatchNorm1d(int(indexes_full_state))

        self.conv1 = nn.Conv1d(input_size, 512, 1)
        self.bn1 = nn.BatchNorm1d(input_size)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.bn2 = nn.BatchNorm1d(512)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.bn3 = nn.BatchNorm1d(256)

        self.conv_final2 = nn.Conv1d(128 + 128, 1, 1)
        self.bn_final = nn.BatchNorm1d(128 + 128)

        self.conv_bias = nn.Conv1d(sim_size, 1, 1)
        self.bn_bias = nn.BatchNorm1d(sim_size)

        self.final_q = nn.Linear(256, 1)

        self.sim_size = sim_size

    def forward(self, x, subset):
        # Compute state representation
        sub = subset.transpose(2, 1).contiguous()
        sub = self.conv1_s(F.relu(self.bn1_s(sub)))
        sub = self.conv2_s(F.relu(self.bn2_s(sub)))
        sub = self.conv3_s(F.relu(self.bn3_s(sub)))
        sub = self.linear_s(F.relu(self.bn_last_s(sub.view(sub.size()[0], -1))))
        sub = sub.unsqueeze(2).repeat(1, 1, x.shape[1])

        bias = self.conv_bias(F.relu(self.bn_bias(x[:, :, -self.sim_size:].transpose(1, 2).contiguous()))).transpose(1,
                                                                                                                     2)

        # Compute action representation
        x = x[:, :, :-self.sim_size].transpose(1, 2).contiguous()
        x = self.conv1(F.relu(self.bn1(x)))
        x = self.conv2(F.relu(self.bn2(x)))
        x = self.conv3(F.relu(self.bn3(x)))

        # Compute Q(s,a)
        out = torch.cat([x, sub], dim=1)
        out = self.conv_final2(self.bn_final(out))
        return (F.sigmoid(bias) * out.transpose(1, 2)).view(out.size()[0], -1)
