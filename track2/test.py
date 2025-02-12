import glob

from engine import *
from dataset import *
from models import *
from torch.autograd import Variable
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Eval Script')
parser.add_argument('--first_model', type=str, default='', help='task 1 model')
parser.add_argument('--second_model', type=str, default='', help='task 2 model')
parser.add_argument('--results', type=str, default='', help='Results location')
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to run the script on: 'cuda' or 'cpu'. ")

opt = parser.parse_args()
model_1 = LightLearningNet()
model_1.load_state_dict(torch.load(opt.first_model, map_location=torch.device('cpu'), weights_only=True))
model_2 = SecondLightResidualNet()
model_2.load_state_dict(torch.load(opt.second_model, map_location=torch.device('cpu'), weights_only=True))

val_images = glob.glob('/Users/kolyszko/Downloads/pirm/validation_hd5/*.h5')

model_1 = model_1.to(opt.device)
model_2 = model_2.to(opt.device)
model_1.eval()
model_2.eval()

print("====> Validation")
for iteration, file in enumerate(val_images, 1):
    basename = os.path.basename(file)[:-3]

    x = h5py.File(file, 'r')['data'][:] / 65535.0
    x_t = torch.from_numpy(x)
    tif_1 = h5py.File(file, 'r')['datatif'][:] / 65535.0
    tif_t = torch.from_numpy(tif_1)

    data = Variable(x_t).to(opt.device)
    data = torch.clamp(data, 0, 1)
    output_1 = torch.clamp(model_1(data), 0, 1)
    output_1 = output_1.detach().cpu().numpy()
    output_1[:, :, 1::2, 1::2] = x[:, :, 1::2, 1::2]
    output_1[:, :, 1::3, 1::3] = x[:, :, 1::3, 1::3]
    output_1 = Variable(torch.from_numpy(output_1)).to(opt.device)

    datatif = Variable(tif_t).to(opt.device)
    datatif = torch.clamp(datatif, 0, 1)
    output = torch.clamp(model_2(output_1, datatif), 0, 1)

    output = output.detach().cpu().numpy()
    output_1 = output_1.detach().cpu().numpy()

    output_g = np.copy(output)
    output_g[:, :, 1::2, 1::2] = x[:, :, 1::2, 1::2]
    output_g[:, :, 1::3, 1::3] = x[:, :, 1::3, 1::3]

    print("===> Image %d" % iteration)

    np.savez('%s/%s.npz' % (opt.results, basename), out=output, out_g=output_g)

