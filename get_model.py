class CRNN_GRU(nn.Module):
    def __init__(self, cnn_output_height, gru_hidden_size, gru_num_layers, num_classes) :
        super(CRNN_GRU, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.norm1 = nn.InstanceNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2)
        self.norm2 = nn.InstanceNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.norm3 = nn.InstanceNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=2)
        self.norm4 = nn.InstanceNorm2d(64)
        self.gru_input_size = cnn_output_height * 64
        self.gru = nn.GRU(self.gru_input_size, gru_hidden_size, gru_num_layers, 
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(gru_hidden_size * 2, num_classes)

    def forward(self, x): # 64,1,32,160
        batch_size = x.shape[0]
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.leaky_relu(out) # 64,32,30,158
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.leaky_relu(out) # 64,32,14,78
        out = self.conv3(out)
        out = self.norm3(out)
        out = F.leaky_relu(out) # 64,64,12,76
        out = self.conv4(out)
        out = self.norm4(out)
        out = F.leaky_relu(out) # 64,64,5,37
        out = out.reshape(batch_size, -1, self.gru_input_size)
        out, _ = self.gru(out) # 64, 37, 256
        out = torch.stack([F.log_softmax(self.fc(out[i]), dim=-1) for i in range(out.shape[0])])
        return out
