import torch
from torch import nn


class LSTM(nn.Module) :
    def __init__(self, opt) :
        # Inheritance
        super(LSTM, self).__init__()
        
        # Initialize Variable
        self.opt = opt

        # Create LSTM Layer Instance
        self.lstm = nn.LSTM(opt.hid_channels, opt.hid_channels, num_layers=opt.num_layer, bidirectional=False, batch_first=True, dropout=opt.p if opt.num_layer != 1 else 0)
        self.bilstm = nn.LSTM(opt.hid_channels, opt.hid_channels//2, num_layers=opt.num_layer, bidirectional=True, batch_first=True, dropout=opt.p if opt.num_layer != 1 else 0)

        # Create FC Layer Instance
        self.input2lstm = nn.Linear(opt.in_channels, opt.hid_channels)
        self.input2bilstm = nn.Linear(opt.in_channels, opt.hid_channels)
        self.input2output = nn.Linear(opt.in_channels, opt.hid_channels)
        self.fc0 = nn.Linear(opt.hid_channels*2, opt.hid_channels, bias=False)
        self.fc1 = nn.Linear(opt.hid_channels, opt.hid_channels, bias=False)
        self.fc2 = nn.Linear(opt.hid_channels, opt.out_channels)
        
        # Create Layer Normalization Layer Instance
        self.norm0 = nn.LayerNorm(opt.hid_channels)
        self.norm1 = nn.LayerNorm(opt.hid_channels)

        # Create Activation Layer Instance
        self.act = nn.ReLU(inplace=True)

    def forward(self, input) :
        lstm_input, bilstm_input = self.input2lstm(input), self.input2bilstm(input)
        
        lstm_h0 = torch.zeros(self.opt.num_layer, lstm_input.size(0), self.opt.hid_channels).to(input.device)
        lstm_c0 = torch.zeros(self.opt.num_layer, lstm_input.size(0), self.opt.hid_channels).to(input.device)
        
        bilstm_h0 = torch.zeros(self.opt.num_layer*2, bilstm_input.size(0), self.opt.hid_channels//2).to(input.device)
        bilstm_c0 = torch.zeros(self.opt.num_layer*2, bilstm_input.size(0), self.opt.hid_channels//2).to(input.device)
        
        lstm_output, _ = self.lstm(lstm_input, (lstm_h0, lstm_c0))
        bilstm_output, _ = self.bilstm(bilstm_input, (bilstm_h0, bilstm_c0))

        output = self.norm0(self.act(self.fc0(torch.cat([lstm_output, bilstm_output], dim=-1))))
        output = self.norm1(self.act(self.fc1(output))) + self.input2output(input)
        output = self.fc2(output)

        return output