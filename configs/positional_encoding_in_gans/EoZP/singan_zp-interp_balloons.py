_base_ = ['../../singan/singan_balloons.py']
model = dict(
    type='PESinGAN',
    generator=dict(
        type='SinGANMSGeneratorPE', interp_pad=False, noise_with_pad=False))

train_cfg = dict(fixed_noise_with_pad=True)

#dist_params = dict(backend='nccl', port=23019)