import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
      }


class VGG(nn.Module):
    def __init__(self, vgg_name):

        super(VGG, self).__init__()
        self.vgg_name = vgg_name
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)

        out = self.classifier(out)

        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        print('%s Model Successfully Built \n' % self.vgg_name)

        return nn.Sequential(*layers)



# As an example of the above, here is VGG16 hard-coded, which is easier to read.

class VGG16(nn.Module):
    def __init__(self):

        super(VGG16, self).__init__()

        self.conv1a = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1a_bn = nn.BatchNorm2d(64)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv1b_bn = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2a_bn = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv2b_bn = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3a_bn = nn.BatchNorm2d(256)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3b_bn = nn.BatchNorm2d(256)
        self.conv3c = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3c_bn = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4a = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4a_bn = nn.BatchNorm2d(512)
        self.conv4b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4b_bn = nn.BatchNorm2d(512)
        self.conv4c = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4c_bn = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5a = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5a_bn = nn.BatchNorm2d(512)
        self.conv5b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5b_bn = nn.BatchNorm2d(512)
        self.conv5c = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5c_bn = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=1, stride=1)

        self.classifier = nn.Linear(512, 100)

        self.relu = nn.ReLU()

        print('VGG16 Model Successfully Built \n')


    def forward(self, x):
        x = self.relu(self.conv1a_bn(self.conv1a(x)))
        x = self.relu(self.conv1b_bn(self.conv1b(x)))
        x = self.pool1(x)


        x = self.relu(self.conv2a_bn(self.conv2a(x)))
        x = self.relu(self.conv2b_bn(self.conv2b(x)))
        x = self.pool2(x)


        x = self.relu(self.conv3a_bn(self.conv3a(x)))
        x = self.relu(self.conv3b_bn(self.conv3b(x)))
        x = self.relu(self.conv3c_bn(self.conv3c(x)))
        x = self.pool3(x)


        x = self.relu(self.conv4a_bn(self.conv4a(x)))
        x = self.relu(self.conv4b_bn(self.conv4b(x)))
        x = self.relu(self.conv4c_bn(self.conv4c(x)))
        x = self.pool4(x)

        x = self.relu(self.conv5a_bn(self.conv5a(x)))
        x = self.relu(self.conv5b_bn(self.conv5b(x)))
        x = self.relu(self.conv5c_bn(self.conv5c(x)))
        x = self.pool5(x)

        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)
        out = self.classifier(x)

        return out


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VGG('VGG16')
    print(model)