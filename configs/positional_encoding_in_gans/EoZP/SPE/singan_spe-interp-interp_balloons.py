_base_ = ['../../../singan/singan_balloons.py']

embedding_dim = 4
num_scales = 8  # start from zero

model = dict(
    type='PENFSinGAN',
    generator=dict(
        type='SinGANMSGeneratorPENF',
        num_scales=num_scales,
        padding=0,
        first_stage_in_channels=embedding_dim * 2,
        positional_encoding=dict(
            type='SPE',
            embedding_dim=embedding_dim,
            padding_idx=0,
            init_size=512,
            div_half_dim=False,
            center_shift=200)),
    discriminator=dict(num_scales=num_scales),
)

train_cfg = dict(
    first_fixed_noises_ch=embedding_dim * 2,
    noise_rs_mode='interp',
    prev_rs_mode='interp',
)