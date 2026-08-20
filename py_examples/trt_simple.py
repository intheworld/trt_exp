import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

with trt.Builder(TRT_LOGGER) as builder, \
     builder.create_builder_config() as config, \
     builder.create_network(0) as network:

    input_tensor = network.add_input(
        name="input",
        dtype=trt.float32,
        shape=(1, 3, 224, 224),
    )

    pool = network.add_pooling_nd(
        input=input_tensor,
        type=trt.PoolingType.MAX,
        window_size=(2, 2),
    )
    pool.stride_nd = (2, 2)

    output = pool.get_output(0)
    output.name = "output"
    network.mark_output(output)

    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, 1 << 30
    )

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("TensorRT engine 构建失败")

    with open("model_python_trt.engine", "wb") as f:
        f.write(serialized_engine)