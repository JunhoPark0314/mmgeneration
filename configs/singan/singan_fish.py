_base_ = [
    './singan_general.py'
]
num_scales = 10  # start from zero
model = dict(
    generator=dict(num_scales=num_scales),
    discriminator=dict(num_scales=num_scales))

total_iters = 22000

data = dict(
    train=dict(img_path='./data/singan/fish.png', min_size=25, max_size=300))