_base_ = [
    './singan_general.py'
]

num_scales = 8  # start from zero
model = dict(
    generator=dict(num_scales=num_scales),
    discriminator=dict(num_scales=num_scales))

data = dict(train=dict(img_path='./data/singan/balloons.png'))

total_iters = 18000