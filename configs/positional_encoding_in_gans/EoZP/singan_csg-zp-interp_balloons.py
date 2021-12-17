_base_ = ['../../singan/singan_balloons.py']

num_scales = 8  # start from zero
model = dict(
    type='PESinGAN',
    generator=dict(
        type='SinGANMSGeneratorPE',
        num_scales=num_scales,
        padding=1,
        pad_at_head=True,
        interp_pad=False,
        noise_with_pad=False,
        first_stage_in_channels=2,
        positional_encoding=dict(type='CSG')),
    discriminator=dict(num_scales=num_scales))

train_cfg = dict(first_fixed_noises_ch=2)