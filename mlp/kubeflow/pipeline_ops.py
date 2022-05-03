def set_gpu_limit(num_gpus, gpu_type='nvidia'):
  def _set_gpu_limit(container_op):
    container_op.set_gpu_limit(num_gpus, gpu_type)
  return _set_gpu_limit


def set_memory_request_and_limits(memory_request, memory_limit):
  def _set_memory_request_and_limits(task):

    return (
        task.container.set_memory_request(memory_request)
            .set_memory_limit(memory_limit)
    )

  return _set_memory_request_and_limits
