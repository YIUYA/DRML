from torch import nn
#from Region_Layer import RegionLayer
from torch.autograd import Variable
import numpy as np
import torch



class RegionLayer(nn.Module):
    def __init__(self, in_channels, grid=(8, 8)):
        super(RegionLayer, self).__init__()

        self.in_channels = in_channels
        self.grid = grid

        self.region_layers = dict()

        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                module_name = 'region_conv_%d_%d' % (i, j)
                self.region_layers[module_name] = nn.Sequential(
                    nn.BatchNorm2d(self.in_channels),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels,
                              kernel_size=3, stride=1, padding=1)
                )
                self.add_module(name=module_name, module=self.region_layers[module_name])

    def forward(self, x):
        """
        :param x:   (b, c, h, w)
        :return:    (b, c, h, w)
        """

        batch_size, _, height, width = x.size()

        input_row_list = torch.split(x, split_size_or_sections=(height//self.grid[0]), dim=2)
        output_row_list = []

        for i, row in enumerate(input_row_list):
            input_grid_list_of_a_row = torch.split(row, split_size_or_sections=width//self.grid[1], dim=3)
            output_grid_list_of_a_row = []

            for j, grid in enumerate(input_grid_list_of_a_row):
                module_name = 'region_conv_%d_%d' % (i, j)
                grid = self.region_layers[module_name](grid.contiguous()) + grid
                output_grid_list_of_a_row.append(grid)

            output_row = torch.cat(output_grid_list_of_a_row, dim=3)
            output_row_list.append(output_row)

        output = torch.cat(output_row_list, dim=2)

        return output

class DRML(nn.Module):
    def __init__(self, class_number=8):
        super(DRML, self).__init__()
        self.class_number = class_number

        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=11, stride=1),
            RegionLayer(in_channels=32, grid=(8, 8)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LocalResponseNorm(5),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=8, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=8, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=16*27*27, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=class_number)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    
    net = DRML()
    print(net)
