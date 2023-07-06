import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 16#128
  training.n_epochs = 20
  training.snapshot_freq = 5
  training.log_freq = 50
  training.eval_freq = 5000
  ## produce samples at each snapshot.
  training.snapshot_sampling = False
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = False
  training.random_crop_size = 0

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.16

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.batch_size = 128
  evaluate.enable_sampling = False

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'XR'
  data.dataset_name = 'bham_gcmx-4x_12em_psl-temp4th-vort4th_eqvt_random-season'
  data.image_size = 64
  data.random_flip = False
  data.centered = False
  data.uniform_dequantization = False
  data.input_transform_key = "stan"
  data.target_transform_key = "sqrturrecen"

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_min = 0.01
  model.sigma_max = 50
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'
  model.map_features = 0 # DEPRECATED, use loc_spec_channels
  model.loc_spec_channels = 8

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config
