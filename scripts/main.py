from models import *
from engine import *
from dataset import *
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser(description='Super Resolution')
parser.add_argument('--model', type=str, default='2', help='model id')
parser.add_argument('--batchSize', type=int, default='32', help='Training batch size')
parser.add_argument("--nEpochs", type=int, default=600, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate. Default=0.001")
parser.add_argument('--save_path', type=str,
                    default='/Users/kolyszko/PycharmProjects/SuperResolution/models/model_1.pt', help="Path to model checkpoint")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to run the script on: 'cuda' or 'cpu'. ")


def main():
    global model
    opt = parser.parse_args()
    print(opt)

    print("===> Loading data")
    train_set = SRDataset(input_transform=get_input_transforms(), target_transform=get_target_transforms())
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)

    valid_set = SRDataset(training=False, input_transform=get_input_transforms(), target_transform=get_target_transforms())
    valid_loader = DataLoader(dataset=valid_set, batch_size=opt.batchSize, shuffle=True)

    print("===> Building model")
    if opt.model == '1':
        model = LightLearningNet()
    elif opt.model == '2':
        model = ResidualLearningNet()
    model = model.to(opt.device)
    mrae = MRAELoss()
    sid = SIDLoss()

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-5)

    print("===> Starting Training")
    train(train_loader,
          valid_loader,
          model,
          opt.nEpochs,
          optimizer,
          opt.device,
          opt.save_path,
          mrae,
          sid,
          opt.lr)


if __name__ == "__main__":
    main()







