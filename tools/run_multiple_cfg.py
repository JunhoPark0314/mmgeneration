import argparse
import json
import os

parser = argparse.ArgumentParser("Multiple cfg runner")
parser.add_argument("--cfg-json", type=str, default="experiments/EoZP_ballons.json")
parser.add_argument("--log-file", type=str, default="log.txt")

args = parser.parse_args()
with open(args.cfg_json) as cf:
	exp_list = json.load(cf)

template_script = "python -m torch.distributed.launch --nproc_per_node  2 --master_port 33132 \
					tools/train.py {cfg_file} --work-dir {work_dir} \
					--cfg-options {cfg_opt} --launcher pytorch"

WORK_ROOT = "output"
unique_name = args.cfg_json.split('.')[0]
error_list = []
cfg_name = os.path.basename(args.cfg_json).split('.')[0]

for exp_dict in exp_list:
	exp_name = exp_dict["exp_name"]
	work_dir = os.path.join(WORK_ROOT, cfg_name, exp_name)

	exp_end = os.path.isdir(os.path.join(work_dir, "final"))

	if exp_end:
		continue

	cfg_file = exp_dict["cfg_file"]
	cfg_opt = str.join(' ',["{}={}".format(k, v) for k, v in exp_dict["cfg_opt"].items()])
	curr_script = template_script.format(cfg_file=cfg_file, work_dir=work_dir, cfg_opt=cfg_opt)

	try:
		os.system(curr_script)
	except Exception as e:
		print(str(e))
		exp_name = exp_dict["exp_name"]
		error_list.append(exp_name)

log_file_path = os.path.join(WORK_ROOT, unique_name, args.log_file)
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
with open(log_file_path, "w") as lf:
	lf.write(str.join('\n', error_list))