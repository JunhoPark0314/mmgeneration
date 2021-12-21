_base_ = ['../../../singan/singan_colusseum.py']
model = dict(
    type='PENFSinGAN',
    generator=dict(
        type='SinGANMSGeneratorPENF'))

train_cfg = dict(
    noise_rs_mode='interp',
    prev_rs_mode='interp',
)