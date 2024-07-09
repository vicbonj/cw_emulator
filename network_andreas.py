from torch import nn


class Emulator(nn.Module):
    def __init__(self, n_params, pix_size=256, filters=8):
        super(Emulator, self).__init__()

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
        

class Test_CNN(nn.Module):
    def __init__(self, n_channels, filters=8):
        super(Test_CNN, self).__init__()

        self.dense1 = nn.Linear(n_channels, 16)
        self.dense2 = nn.Linear(16, 128)
        self.dense3 = nn.Linear(128, 10*filters*8)

        self.act = nn.SELU()
        self.transp = nn.Upsample(scale_factor=2, mode='linear')
        #self.upconv1 = nn.ConvTranspose1d(filters*8, filters*4, 2)
        self.conv1 = nn.Conv1d(filters*8, filters*4, 3, padding=int((3-1)/2), padding_mode='reflect')
        #self.upconv2 = nn.ConvTranspose1d(filters*4, filters*2, 2)
        self.conv2 = nn.Conv1d(filters*4, filters*2, 3, padding=int((3-1)/2), padding_mode='reflect')
        #self.upconv3 = nn.ConvTranspose1d(filters*2, filters, 2)
        self.conv3 = nn.Conv1d(filters*2, filters, 3, padding=int((3-1)/2), padding_mode='reflect')
        self.conv4 = nn.Conv1d(filters, 1, 1)

    def forward(self, x):

        x = self.act(self.dense1(x))
        #print(x.size())
        x = self.act(self.dense2(x))
        #print(x.size())
        x = self.act(self.dense3(x))#[:,:,None]
        #print(x.size())
        x = x.reshape(x.shape[0], 64, 10)
        #print(x.size())
        x = self.act(self.conv1(self.transp(x)))
        #print(x.size())
        x = self.act(self.conv2(self.transp(x)))
        #print(x.size())
        x = self.act(self.conv3(self.transp(x)))
        #print(x.size())
        x = self.conv4(x)
        #print(x.size())

        return x[:,0]
