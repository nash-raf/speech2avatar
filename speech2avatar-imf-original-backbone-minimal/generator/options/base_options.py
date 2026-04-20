import os, argparse, json

class BaseOptions():
	def parse(self):
		parser = argparse.ArgumentParser()
		self.parser = self.initialize(parser)
		self.opt = self.parser.parse_args()
		return self.opt

	def initialize(self, parser):
		parser.add_argument('--pretrained_dir', type=str, default='./checkpoints')
		parser.add_argument('--seed', default=42, type=int)
		parser.add_argument('--fix_noise_seed', action='store_true')

		# video
		parser.add_argument('--input_size', type=int, default=256, help='input image size')
		parser.add_argument('--input_nc', type=int, default=3, help='input image channel')        
		parser.add_argument('--fps', type=float, default=25.)

		# audio
		parser.add_argument('--sampling_rate', type=int, default=16000)
		parser.add_argument('--audio_marcing', type=int, default=2, help='number of adjacent frames. For value v, t -> [t-v, ..., t, ..., t+v]')        
		parser.add_argument('--wav2vec_sec', default=2, type=float, help='window length L (seconds), 50 frames')
		parser.add_argument('--wav2vec_model_path', default='./checkpoints/wav2vec2-base-960h')
		parser.add_argument('--attention_window', default=5, type=int, help='attention window size, e.g., if 1, attend frames of t-1, t, t+1 for frame t')

		parser.add_argument('--only_last_features', type=bool, default="true")
		parser.add_argument('--average_emotion', action='store_true', help='averaging emotion or not.')

		# dropout
		parser.add_argument('--audio_dropout_prob', default=0.1, type=float)
		parser.add_argument('--ref_dropout_prob', default=0.1, type=float)
		parser.add_argument('--emotion_dropout_prob', default=0.1, type=float)

		# model Hyper Parameters
		parser.add_argument('--style_dim', type=int, default=512, help='w latent dimension')
		parser.add_argument('--dim_a', type=int, default=512, help='audio dimension')
		parser.add_argument('--dim_h', type=int, default=512, help='hidden dimension')
		parser.add_argument('--dim_e', type=int, default=7, help='emotion dimension')
		parser.add_argument("--dim_motion", type=int, default=32)
		parser.add_argument("--dim_c", type=int, default=32)
		parser.add_argument('--dim_w', type=int, default=32, help='face dimension')

		# option for FMT
		parser.add_argument('--fmt_depth', default=8, type=int)
		parser.add_argument('--aux_head_depth', default=4, type=int)
		parser.add_argument('--num_heads', default=8, type=int)
		parser.add_argument('--mlp_ratio', default=4.0, type=float)
		parser.add_argument('--drop_path_rate', default=0.1, type=float, help='Stochastic depth rate')
		parser.add_argument('--no_learned_pe', action='store_true')
		parser.add_argument('--num_prev_frames', type=int, default=10)
		parser.add_argument('--max_grad_norm', default=1, type=float, help='max grad norm for training transformers')
		parser.add_argument('--num_time_tokens', default=4, type=int)
		parser.add_argument('--num_cfg_tokens', default=4, type=int)
		parser.add_argument('--num_interval_tokens', default=2, type=int)

		# iMF schedule / objective
		parser.add_argument('--P_mean', default=-0.4, type=float)
		parser.add_argument('--P_std', default=1.0, type=float)
		parser.add_argument('--data_proportion', default=0.5, type=float)
		parser.add_argument('--cfg_beta', default=1.0, type=float)
		parser.add_argument('--norm_p', default=1.0, type=float)
		parser.add_argument('--norm_eps', default=0.01, type=float)
		parser.add_argument('--lambda_vel', default=0.05, type=float,
							help='Temporal velocity smoothness loss weight')
		parser.add_argument('--prev_dropout_prob', default=0.5, type=float,
							help='Dropout probability for previous-context signals')

		parser.add_argument('--ode_atol', default=1e-5, type=float)
		parser.add_argument('--ode_rtol', default=1e-5, type=float)
		parser.add_argument('--nfe', default=1, type=int,
							help='Number of finite iMF solver steps')
		parser.add_argument('--torchdiffeq_ode_method', default='euler',
							help='ODE solver')
		parser.add_argument('--a_cfg_scale', default=3, type=float,
							help='audio classifier-free guidance (vector field) scale')             
		parser.add_argument('--swin_res_threshold', type=int, default=128, help='Resolution threshold to switch to Swin Attention.')
		parser.add_argument('--window_size', type=int, default=8, help='Window size for Swin Attention.')

		return parser


	def print_options(self):
		"""Print and save options

		It will print both current options and default values(if different).
		It will save options into a text file / [checkpoints_dir] / opt.txt
		"""
		message = ''
		message += '----------------- Options ---------------\n'
		for k, v in sorted(vars(self.opt).items()):
			comment = ''
			default = self.parser.get_default(k)
			if v != default:
				comment = '\t[default: %s]' % str(default)
			message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
		message += '----------------- End -------------------'
		print(message)


def save_options(opt, save_path):
	with open(save_path, 'wt') as f:
		json.dump(vars(opt), f, indent=4)


def load_options(opt, load_path):
	with open(load_path, 'rt') as f:
		_update = json.loads(f)
	opt.update(_update)
	return opt
