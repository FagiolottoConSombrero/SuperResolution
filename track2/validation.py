import glob
import matplotlib.pyplot as plt
from dataset import *
from models import *
from torch.autograd import Variable
import argparse
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

parser = argparse.ArgumentParser(description='Eval Script')
parser.add_argument('--first_model', type=str, default='', help='task 1 model')
parser.add_argument('--second_model', type=str, default='', help='task 2 model')
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device to run the script on: 'cuda' or 'cpu'. ")


def MSE(gt, rc):
    return np.mean((gt - rc) ** 2)


def PSNR(gt, rc):
    mse = MSE(gt, rc)
    pmax = 1
    return 10 * np.log10(pmax **1 / mse)


def MRAE(gt, rc):
    return np.mean(np.abs(gt - rc) / (gt + 1e-3))


def SID(gt, rc):
    epsilon = 1e-3  # valore minimo per evitare divisioni per zero
    max_val = 1.1998

    # Calcola le somme lungo l'asse delle bande e applica il clipping per evitare divisioni per 0
    gt_sum = np.clip(np.sum(gt, axis=0, keepdims=True), a_min=epsilon, a_max=None)
    rc_sum = np.clip(np.sum(rc, axis=0, keepdims=True), a_min=epsilon, a_max=None)

    # Normalizza gt e rc
    gt_norm = gt / gt_sum
    rc_norm = rc / rc_sum

    # Calcola i rapporti; qui gestiamo esplicitamente il caso 0/0
    ratio1 = np.where((gt_norm == 0) & (rc_norm == 0), 1.0, gt_norm / rc_norm)
    ratio2 = np.where((gt_norm == 0) & (rc_norm == 0), 1.0, rc_norm / gt_norm)

    # Applica il clipping prima del logaritmo per evitare valori fuori dall'intervallo desiderato
    ratio1_clipped = np.clip(ratio1, a_min=epsilon, a_max=max_val)
    ratio2_clipped = np.clip(ratio2, a_min=epsilon, a_max=max_val)

    divergence = (gt_norm * np.log(ratio1_clipped) +
                  rc_norm * np.log(ratio2_clipped))

    # Somma le divergenze lungo l'asse delle bande e fai la media sui restanti pixel
    return np.mean(np.sum(divergence, axis=0))


def APPSA(gt, rc):
    dot_product = np.sum(gt * rc, axis=0)
    norm_gt = np.sqrt(np.sum(gt ** 2, axis=0))
    norm_pred = np.sqrt(np.sum(rc ** 2, axis=0))

    cos_theta = np.clip(dot_product / (norm_gt * norm_pred + 1e-6), -1, 1)
    spectral_agle = np.arccos(cos_theta)
    return np.mean(spectral_agle)


def SSIM(gt, rc):
    ssim_values = []
    for i in range(gt.shape[0]):
        ssim_value = compare_ssim(
                rc[i],
                gt[i],
                data_range=rc[i].max() - rc[i].min(),  # Explicit data range
                channel_axis=0  # Specify the channel axis
            )
        ssim_values.append(ssim_value)
    return np.mean(ssim_values)


# First element is the ground truth, second is the prediction
measures = {
    'APPSA': APPSA,
    'SID': SID,
    'PSNR': PSNR,
    'SSIM': SSIM,
    'MSE': MSE,
    'MRAE': MRAE
}

opt = parser.parse_args()
model_1 = LightLearningNet()
model_1.load_state_dict(torch.load(opt.first_model, map_location=torch.device('cpu'), weights_only=True))
model_2 = SecondLightResidualNet()
model_2.load_state_dict(torch.load(opt.second_model, map_location=torch.device('cpu'), weights_only=True))

val_images = glob.glob('/Users/kolyszko/Downloads/pirm/test_data/*.h5')

model_1 = model_1.to(opt.device)
model_2 = model_2.to(opt.device)
model_1.eval()
model_2.eval()
summed_measures = None

print("====> Validation")
for iteration, h5pyfilename in enumerate(val_images, 1):

    image = h5py.File(h5pyfilename, 'r')['data'][:] / 65535.0
    image_t = torch.from_numpy(image)
    image_1 = h5py.File(h5pyfilename, 'r')['datatif'][:] / 65535.0
    image_tif = torch.from_numpy(image_1)
    hrimg = h5py.File(h5pyfilename, 'r')['hrdata'][:] / 65535.0
    image_hr = torch.from_numpy(hrimg)

    data = Variable(image_t).to(opt.device)
    data = torch.clamp(data, 0, 1)
    output1 = torch.clamp(model_1(data), 0, 1)
    output1 = output1.detach().cpu().numpy()
    output1[:, :, 1::2, 1::2] = image[:, :, 1::2, 1::2]
    output1[:, :, 1::3, 1::3] = image[:, :, 1::3, 1::3]
    output1 = Variable(torch.from_numpy(output1)).to(opt.device)

    datatif = Variable(image_tif).to(opt.device)
    datatif = torch.clamp(datatif, 0, 1)
    output = torch.clamp(model_2(output1, datatif), 0, 1)

    output = output.detach().cpu().numpy()
    output1 = output1.detach().cpu().numpy()

    output[:, :, 1::2, 1::2] = image[:, :, 1::2, 1::2]
    output[:, :, 1::3, 1::3] = image[:, :, 1::3, 1::3]
    output1 = output1[0] #* 65535.0
    output = output[0] #* 65535.0
    gt = image_hr[0].numpy()  # / 65535.0

    '''gt_first_band = gt[0, :, :]
    output_first_band = output1[0, :, :]

    # Visualizza le immagini affiancate
    plt.figure(figsize=(10, 5))

    # Plot per gt
    plt.subplot(1, 2, 1)
    plt.imshow(gt_first_band.T, cmap='gray')
    plt.title("GT - Prima banda")
    plt.axis('off')

    # Plot per output
    plt.subplot(1, 2, 2)
    plt.imshow(output_first_band.T, cmap='gray')
    plt.title("Output - Prima banda")
    plt.axis('off')

    plt.tight_layout()
    plt.show()'''


    hr_measures = {k: np.array(func(gt, output)) for (k, func) in measures.items()}
    origin_measures = {k: np.array(func(gt, output1)) for (k, func) in measures.items()}

    print("===> Image %d" % iteration)

    if summed_measures is None:
        origin_summed = origin_measures
        summed_measures = hr_measures
    else:
        origin_summed = {k: v + origin_measures[k] for (k, v) in origin_summed.items()}
origin_summed = {k: v / 10 for (k, v) in origin_summed.items()}
print('Average Measures')
print(origin_summed)

