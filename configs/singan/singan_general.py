_base_ = [
    '../_base_/models/singan/singan.py', '../_base_/datasets/singan.py',
    '../_base_/default_runtime.py'
]

train_cfg = dict(
    noise_weight_init=0.1,
    iters_per_scale=2000,
)

optimizer = None
lr_config = None
checkpoint_config = dict(by_epoch=False, interval=2000, max_keep_ckpts=3)

custom_hooks = [
    dict(
        type='MMGenVisualizationHook',
        output_dir='visual',
        interval=500,
        bgr2rgb=True,
        res_name_list=['fake_imgs', 'recon_imgs', 'real_imgs']),
    dict(
        type='PickleDataHook',
        output_dir='pickle',
        interval=-1,
        after_run=True,
        data_name_list=['noise_weights', 'fixed_noises', 'curr_stage'])
]

