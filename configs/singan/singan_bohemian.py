_base_ = [
    './singan_general.py'
]


num_scales = 10  # start from zero
model = dict(
    generator=dict(num_scales=num_scales),
    discriminator=dict(num_scales=num_scales))

data = dict(
    train=dict(
        img_path='./data/singan/bohemian.png', min_size=25, max_size=500))

total_iters = 22000