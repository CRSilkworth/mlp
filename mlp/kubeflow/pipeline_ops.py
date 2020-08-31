def set_gpu_limit(num_gpus, gpu_type='nvidia'):
  def _set_gpu_limit(container_op):
    container_op.set_gpu_limit(num_gpus, gpu_type)
  return _set_gpu_limit
