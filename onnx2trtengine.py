import tensorrt as trt
import time
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)  # Set the logging level


def create_builder() -> trt.Builder:
    """
    Create and return a TensorRT builder instance.

    Returns:
        out (trt.Builder):
            TensorRT builder instance
    """
    return trt.Builder(TRT_LOGGER)


def parse_onnx(builder: trt.Builder, onnx_path: str) -> trt.INetworkDefinition:
    """
    Parse an ONNX file to create a TensorRT network.

    Args:
        builder (trt.Builder):
            TensorRT builder instance
        onnx_path (str):
            Path to the ONNX file with the model architecture and weights
    Returns:
        out (trt.INetworkDefinition):
            TensorRT network definition
    """
    # Create the builder, network, and parser
    # TODO: Do not make it explicit batch size, make it dynamic batch size
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    # Create a network with explicit batch flag enabled
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse the ONNX file
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise Exception(f"Failed to parse the ONNX file: {onnx_path}")
        print(f"Successfully parsed the ONNX file: {onnx_path}")

    # Print the network layers
    print(f"Number of layers in the network: {network.num_layers}")
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        print(f"Layer {i}: {layer.name}, type: {layer.type}, inputs: {layer.num_inputs}, outputs: {layer.num_outputs}")

    # Get the input binding name and expected input shape
    input_tensor = network.get_input(0)  # Assuming single input -- TODO: Handle (rare) edge case of multiple inputs
    input_name = input_tensor.name
    input_shape = input_tensor.shape

    print(f"Input binding name after parsing: {input_name}")
    print(f"Input shape after parsing: {input_shape}")

    return network


def create_engine(builder: trt.Builder, 
                  network: trt.INetworkDefinition,
                  max_mem: int = 1 * (1024 ** 3),
                  min_batch_size: int = 1,
                  opt_batch_size: int = 4,
                  max_batch_size: int = 8
                  ) -> trt.ICudaEngine:
    """
    Convert the parsed ONNX network to a TensorRT engine.
    
    Args:
        builder (trt.Builder):
            TensorRT builder instance
        network (trt.INetworkDefinition):
            TensorRT network definition parsed from the ONNX model
        max_mem (int):
            Maximum memory that the engine can use while building (in bytes)
        min_batch_size (int):
            Minimum batch size that the engine should support
        opt_batch_size (int):
            Optimum batch size that the engine should support
        max_batch_size (int):
            Maximum batch size that the engine should support
    Returns:
        out (trt.ICudaEngine):
            TensorRT engine
    """
    input_tensor = network.get_input(0) # Assuming single input -- TODO: Handle (rare) edge case of multiple inputs
    if network.num_inputs != 1:
        raise ValueError(f"Expected a single input in the network, but got: {network.num_inputs} inputs.")
    input_shape = input_tensor.shape
    input_name = input_tensor.name

    output_tensor = network.get_output(0) # Assuming single output -- TODO: Handle (rare) edge case of multiple outputs
    if network.num_outputs != 1:
        raise ValueError(f"Expected a single output in the network, but got: {network.num_outputs} outputs.")
    output_shape = output_tensor.shape
    output_name = output_tensor.name


    # Check if batch size (first dimension) is dynamic
    if input_shape[0] != -1:
        raise ValueError(f"The ONNX model's batch size is fixed (not dynamic). "
                         f"Expected batch size to be dynamic (-1), but instead got input shape: {input_shape}.")

    # Create a builder configuration
    config = builder.create_builder_config()

    # Create an optimization profile for dynamic dimensions
    profile = builder.create_optimization_profile()
    min_shape = (min_batch_size,) + input_shape[1:]  # Minimum input shape you plan to support
    opt_shape = (opt_batch_size,) + input_shape[1:]  # Optimum input shape you plan to support
    max_shape = (max_batch_size,) + input_shape[1:]  # Maximum input shape you plan to support
    # Set the profile shapes with the correct input tensor name
    # TODO: Handle (rare) edge case of multiple inputs
    profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
    
    # Do the same for the output tensor
    # min_shape = (min_batch_size,) + output_shape[1:]  # Minimum output shape you plan to support
    # opt_shape = (opt_batch_size,) + output_shape[1:]  # Optimum output shape you plan to support
    # max_shape = (max_batch_size,) + output_shape[1:]  # Maximum output shape you plan to support
    # profile.set_shape(output_name, min=min_shape, opt=opt_shape, max=max_shape)

    # Set the optimization profile for the builder configuration
    config.add_optimization_profile(profile)

    # Enable FP16 mode if supported
    # TODO: Enable this after testing, and change inference code to use FP16 / FP32 dynamically
    # if builder.platform_has_fast_fp16:
    #     config.set_flag(trt.BuilderFlag.FP16)

    # Enable INT8 mode if a calibration file is available
    # config.set_flag(trt.BuilderFlag.INT8)

    # Set the maximum workspace size using the new method
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_mem)

    # Enable sparsity support if your model has sparse weights
    # config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)

    # Enable profiling to identify bottlenecks
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    # Enable GPU fallback if the operation cannot be supported by TensorRT layers
    # config.set_flag(trt.BuilderFlag.STRICT_TYPES) # Error: this way of setting strict types does not work in latest tensorrt

    # Measure time for building the engine
    start_time = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    end_time = time.time()
    if serialized_engine is None:
        raise Exception("Failed to build the serialized TensorRT engine!")

    # Print the time taken to build the engine
    print(f"Time taken to build the TensorRT engine: {end_time - start_time:.2f} seconds")

    # Deserialize the engine if you need to work with it in code
    engine = None
    with trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(serialized_engine)
    if engine is None:
        raise Exception("Failed to deserialize the TensorRT engine!")

    return engine


def save_engine(engine: trt.ICudaEngine, out_path: str) -> None:
    # Save the serialized engine to the specified path
    with open(out_path, "wb") as f:
        f.write(engine.serialize())
        print(f"Successfully saved the TensorRT engine to: {out_path}")


def load_engine(engine_file_path):
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    return engine

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True, help="Path to the ONNX file")
    parser.add_argument("--engine", type=str, required=True, help="Path to save the TensorRT engine")
    parser.add_argument("--max_mem", type=int, default=4, help="Maximum memory for the engine in GB")
    args = parser.parse_args()

    onnx_path = args.onnx
    engine_path = args.engine
    max_mem = args.max_mem
    max_mem = max_mem * (1024 ** 3)  # Convert GB to bytes

    # Create the builder instance once
    builder = create_builder()

    # Parse the ONNX model, create the engine, and save it
    network = parse_onnx(builder, onnx_path)
    engine = create_engine(builder, network, max_mem)
    save_engine(engine, engine_path)
