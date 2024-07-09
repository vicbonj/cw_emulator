from torch import nn


class EmulatorNet(nn.Module):
    def __init__(self, n_params, pix_size=256, filters=8):
        super(EmulatorNet, self).__init__()

        self.filters = filters
        self.pix_size = pix_size
        self.dense1 = nn.Linear(n_params, 16)
        self.dense2 = nn.Linear(16, 128)
        self.dense3 = nn.Linear(128, int((pix_size/2**5)**2*filters*32))

        self.act = nn.SELU()
        self.transp = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv0 = nn.Conv2d(filters*32, filters*16, 3, bias=False, padding=int((3-1)/2), padding_mode='reflect')
        self.conv1 = nn.Conv2d(filters*16, filters*8, 3, bias=False, padding=int((3-1)/2), padding_mode='reflect')
        self.conv2 = nn.Conv2d(filters*8, filters*4, 3, bias=False, padding=int((3-1)/2), padding_mode='reflect')
        self.conv3 = nn.Conv2d(filters*4, filters*2, 3, bias=False, padding=int((3-1)/2), padding_mode='reflect')
        self.conv4 = nn.Conv2d(filters*2, filters, 3, bias=False, padding=int((3-1)/2), padding_mode='reflect')
        self.conv5 = nn.Conv2d(filters, 1, 1)

    def forward(self, x):
        x = self.act(self.dense1(x))
        x = self.act(self.dense2(x))
        x = self.act(self.dense3(x))

        x = x.reshape(x.shape[0], self.filters*32, int(self.pix_size/2**5), int(self.pix_size/2**5))

        x = self.act(self.conv0(self.transp(x)))
        x = self.act(self.conv1(self.transp(x)))
        x = self.act(self.conv2(self.transp(x)))
        x = self.act(self.conv3(self.transp(x)))
        x = self.act(self.conv4(self.transp(x)))
        x = self.conv5(x)

        return x
