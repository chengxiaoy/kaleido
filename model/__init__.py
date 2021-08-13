from model.base import *
from model.vae import *
from model.beta_vae import *
from model.normalizing_flow_vae import NF_VAE
from model.vq_vae import VQ_VAE
from model.vq_vae2 import VQ_VAE2
from model.vae_gan import VAEGAN
from model.ali import ALI
from model.age import AGE
from model.vee_gan import VEE_GAN
from model.intro_vae import INTRO_VAE
from model.dc_gan import DCGAN
from model.style_gan import StyleGAN

VAE = VanillaVAE
GaussianVAE = VanillaVAE

vae_models = {'VanillaVAE': VanillaVAE, "BetaVAE": BetaVAE, "NF_VAE": NF_VAE, "VQ_VAE": VQ_VAE, "VQ_VAE2": VQ_VAE2,
              "GAN_VAE": VAEGAN, "ALI": ALI, "AGE": AGE, "VEE_GAN": VEE_GAN, "INTRO_VAE": INTRO_VAE, "DCGAN": DCGAN,
              "Style_GAN": StyleGAN}
