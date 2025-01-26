from engine import *
from dataset import *
from torch.utils.data import DataLoader
import argparse
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

parser = argparse.ArgumentParser(description='Super Resolution')
parser.add_argument("--model", default="checkpoint/model_epoch_600.pth", type=str, help="model path")
parser.add_argument("--results", default="results", type=str, help="Result save location")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to run the script on: 'cuda' or 'cpu'. ")


def MSE(gt, rc):
    return np.mean((gt - rc) ** 2)


def PSNR(gt, rc):
    mse = MSE(gt, rc)
    pmax = 65536
    return 20 * np.log10(pmax / np.sqrt(mse + 1e-3))


def MRAE(gt, rc):
    return np.mean(np.abs(gt - rc) / (gt + 1e-3))


def SID(gt, rc):
    N = gt.shape[0]
    err = np.zeros(N)
    for i in range(N):
        err[i] = abs(np.sum(rc[i] * np.log10((rc[i] + 1e-3) / (gt[i] + 1e-3))) +
                     np.sum(gt[i] * np.log10((gt[i] + 1e-3) / (rc[i] + 1e-3))))
    return err.mean()


def APPSA(gt, rc):
    nom = np.sum(gt * rc, axis=0)
    denom = np.linalg.norm(gt, axis=0) * np.linalg.norm(rc, axis=0)

    cos = np.where((nom / (denom + 1e-3)) > 1, 1, (nom / (denom + 1e-3)))
    appsa = np.arccos(cos)

    return np.sum(appsa) / (gt.shape[1] * gt.shape[0])


def SSIM(gt, rc):
    return compare_ssim(gt, rc)


# First element is the ground truth, second is the prediction
measures = {
    'APPSA': APPSA,
    'SID' : SID,
    'PSNR': PSNR,
    'SSIM': compare_ssim,
    'MSE' : MSE,
    'MRAE': MRAE
}

opt = parser.parse_args()
model = torch.load(opt.model)

valid_set = Hdf5Dataset(opt.data_path, training=False, transforms=get_transforms())
valid_loader = DataLoader(dataset=valid_set, batch_size=opt.batchSize, shuffle=True)

model = model.to(opt.device)
summed_measures = None

print("===> Validation")
for iteration, (data, gt) in enumerate(valid_loader, 1):
    data = data.to(opt.device)
    output = model(data)

    output = output.detach().cpu().numpy()
    output = output[0]

    hr_measures = {k: np.array(func(gt, output)) for (k, func) in measures.items()}

    print("===> Image %d" % iteration)
    print(hr_measures)

    if summed_measures is None:
        summed_measures = hr_measures
    else:
        summed_measures = {k: v + hr_measures[k] for (k, v) in summed_measures.items()}

summed_measures = {k: v / len(valid_loader) for (k, v) in summed_measures.items()}
print('Average Measures')
print(summed_measures)