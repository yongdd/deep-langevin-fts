import torch

class AtrNet(torch.nn.Module):
    def __init__(self, dim, in_channels=3, mid_channels = 64, out_channels=1, kernel_size = 3):
        super().__init__()
        padding = (kernel_size-1)//2
        
        if dim == 1:
            self.conv1 = torch.nn.Conv1d(in_channels,  mid_channels, kernel_size, padding=padding, padding_mode='circular')
            self.conv2 = torch.nn.Conv1d(mid_channels, mid_channels, kernel_size, padding=padding, padding_mode='circular')
            self.conv3 = torch.nn.Conv1d(mid_channels, mid_channels, kernel_size, padding=padding*2, padding_mode='circular', dilation=2)
            self.conv4 = torch.nn.Conv1d(mid_channels, mid_channels, kernel_size, padding=padding*4, padding_mode='circular', dilation=4)
            self.conv5 = torch.nn.Conv1d(mid_channels, mid_channels, kernel_size, padding=padding*8, padding_mode='circular', dilation=8)
            self.conv6 = torch.nn.Conv1d(mid_channels, mid_channels, kernel_size, padding=padding*8, padding_mode='circular', dilation=8)
            self.conv7 = torch.nn.Conv1d(mid_channels, out_channels, 1)
            
        elif dim == 2:
            self.conv1 = torch.nn.Conv2d(in_channels,  mid_channels, kernel_size, padding=padding, padding_mode='circular')
            self.conv2 = torch.nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=padding, padding_mode='circular')
            self.conv3 = torch.nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=padding*2, padding_mode='circular', dilation=2)
            self.conv4 = torch.nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=padding*4, padding_mode='circular', dilation=4)
            self.conv5 = torch.nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=padding*8, padding_mode='circular', dilation=8)
            self.conv6 = torch.nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=padding*8, padding_mode='circular', dilation=8)
            self.conv7 = torch.nn.Conv2d(mid_channels, out_channels, 1)
            
        elif dim == 3:
            self.conv1 = torch.nn.Conv3d(in_channels,  mid_channels, kernel_size, padding=padding, padding_mode='circular')
            self.conv2 = torch.nn.Conv3d(mid_channels, mid_channels, kernel_size, padding=padding, padding_mode='circular')
            self.conv3 = torch.nn.Conv3d(mid_channels, mid_channels, kernel_size, padding=padding*2, padding_mode='circular', dilation=2)
            self.conv4 = torch.nn.Conv3d(mid_channels, mid_channels, kernel_size, padding=padding*4, padding_mode='circular', dilation=4)
            self.conv5 = torch.nn.Conv3d(mid_channels, mid_channels, kernel_size, padding=padding*8, padding_mode='circular', dilation=8)
            self.conv6 = torch.nn.Conv3d(mid_channels, mid_channels, kernel_size, padding=padding*8, padding_mode='circular', dilation=8)
            self.conv7 = torch.nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        x = torch.nn.functional.relu(self.conv5(x))
        x = torch.nn.functional.relu(self.conv6(x))
        x = self.conv7(x)
        return x
