_base_ = ['../../../singan/singan_balloons.py']

num_scales = 8  # start from zero
model = dict(
    type='PENFSinGAN',
    generator=dict(
        type='SinGANMSGeneratorPENF',
        num_scales=num_scales,
        padding=0,
        first_stage_in_channels=2,
        positional_encoding=dict(type='CSG')),
    discriminator=dict(num_scales=num_scales))

train_cfg = dict(
    first_fixed_noises_ch=2,
    noise_rs_mode='interp',
    prev_rs_mode='interp',
)