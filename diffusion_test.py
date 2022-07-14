import math
from pathlib import Path

import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from ema_pytorch import EMA
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torchvision import transforms, utils
from tqdm import tqdm

from data_samples import get_beauty_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
resolution = 64
t = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Lambda(lambda X: 2 * X - 1.)
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ]
)
train_dataloader = get_beauty_dataloader(resolution, batch_size, transform=t)
writer = SummaryWriter()

# should be a U-NET model, like PixelCNN++
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    image_size=resolution,
    timesteps=1000,  # number of steps
    loss_type='l1'  # L1 or L2
)
diffusion = diffusion.to(device)
opt = Adam(diffusion.parameters(), lr=1e-4, betas=(0.9, 0.99))


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


dl = cycle(train_dataloader)
train_num_steps = 100000
gradient_accumulate_every = 2
step = 0
ema = EMA(diffusion, beta=0.995, update_every=10)
save_and_sample_every = 1000
num_samples = 25
with tqdm(initial=0, total=train_num_steps) as pbar:
    while step < train_num_steps:
        total_loss = 0.
        for _ in range(gradient_accumulate_every):
            data = next(dl)[0].to(device)
            loss = diffusion(data)
            loss = loss / gradient_accumulate_every
            total_loss += loss.item()
            loss.backward()

        pbar.set_description(f'loss: {total_loss:.4f}')

        opt.step()
        opt.zero_grad()

        ema.to(device)
        ema.update()

        if step != 0 and step % save_and_sample_every == 0:
            ema.ema_model.eval()

            with torch.no_grad():
                milestone = step // save_and_sample_every
                batches = num_to_groups(num_samples, batch_size)
                all_images_list = list(map(lambda n: ema.ema_model.sample(batch_size=n), batches))

            all_images = torch.cat(all_images_list, dim=0)
            utils.save_image(all_images, str(Path("logs/DDPM") / f'sample-{milestone}.png'),
                             nrow=int(math.sqrt(num_samples)))

        step += 1
        pbar.update(1)

print('training complete')
