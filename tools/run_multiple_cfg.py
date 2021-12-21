import argparse
import json
import os
from subprocess import Popen, PIPE

parser = argparse.ArgumentParser("Multiple cfg runner")
parser.add_argument("--cfg-json", type=str, default="experiments/EoZP.json")
parser.add_argument("--log-file", type=str, default="log.txt")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--image", type=str, default="balloons")
parser.add_argument("--PE", type=str, default="IMP")

args = parser.parse_args()
with open(args.cfg_json) as cf:
	exp_list = json.load(cf)

template_script = "python -m torch.distributed.launch --nproc_per_node  2 --master_port 33132 tools/train.py {cfg_file} --work-dir {work_dir} --cfg-options {cfg_opt} --launcher pytorch"

WORK_ROOT = "output"
unique_name = str.join('_', [args.cfg_json.split('.')[0], args.image])
error_list = []
cfg_name = os.path.basename(args.cfg_json).split('.')[0]

for exp_dict in exp_list:
	exp_name = exp_dict["exp_name"].format(image=args.image, PE_low = args.PE.lower())
	work_dir = os.path.join(WORK_ROOT, str.join('_',[cfg_name, args.image]), exp_name)

	exp_end = os.path.isdir(os.path.join(work_dir, "final"))

	if exp_end:
		continue

	cfg_file = exp_dict["cfg_file"].format(image=args.image, PE=args.PE, PE_low=args.PE.lower())

	for k in list(exp_dict["cfg_opt"].keys()):
		if k[0] == '_':
			if args.debug:
				exp_dict["cfg_opt"][k[1:]] = exp_dict["cfg_opt"][k]
			exp_dict["cfg_opt"].pop(k)

	cfg_opt = str.join(' ',["{}={}".format(k, v) for k, v in exp_dict["cfg_opt"].items()])
	curr_script = template_script.format(cfg_file=cfg_file, work_dir=work_dir, cfg_opt=cfg_opt)

	p = Popen(curr_script, stderr=PIPE, shell=True, universal_newlines=True)
	out, error = p.communicate()
	trace_list = error.split('Traceback')
	exp_name = exp_dict["exp_name"].format(image=args.image, PE_low=args.PE.lower())
	error_list.append("{}\n{}\n".format(exp_name, str.join('---------TRACE--------\n', trace_list)))

log_file_path = os.path.join(WORK_ROOT, unique_name, args.PE.lower(), args.log_file)
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
with open(log_file_path, "w") as lf:
	lf.write(str.join('---------------------\n', error_list))