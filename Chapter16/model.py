from torch_snippets import *


class DQNetworkImageSensor(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 32, (8, 8), stride=4),
            nn.Conv2d(32, 64, (4, 4), stride=2),
            nn.Conv2d(64, 128, (3, 3), stride=1),
            nn.AvgPool2d(8),
            nn.Flatten(),
            nn.Linear(1152, 512),
            nn.Linear(512, 9)
        )

        self.lidar_branch = nn.Sequential(
            nn.Conv2d(3, 32, (8, 8), stride=4),
            nn.Conv2d(32, 64, (4, 4), stride=2),
            nn.Conv2d(64, 128, (3, 3), stride=1),
            nn.AvgPool2d(8),
            nn.Flatten(),
            nn.Linear(1152, 512),
            nn.Linear(512, 9)
        )

        self.sensor_branch = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 9)
        )
        
    def forward(self, image, lidar=None, sensor=None):
        x = self.image_branch(image)
        if lidar is None:
            y = 0
        else:
            y = self.lidar_branch(lidar)
        z = self.sensor_branch(sensor)

        return x + y + z
