from typing import List, Optional, Tuple

from huggingface_hub import snapshot_download
import torch
import time
import bench_global_vars

from vllm import LLM, SamplingParams, EngineArgs, LLMEngine, RequestOutput
from vllm.lora.request import LoRARequest

sql_lora_path = "/downloads/adapter/llama3-sql-lora"
chinese_lora_path = "/downloads/adapter/llama3-chinese-lora"
docker_lora_path = "/downloads/adapter/llama3-docker-lora"

def llama3_instruct_prompt(user_prompt: str):
    return (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        + user_prompt
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def create_test_prompts() -> List[Tuple[str, SamplingParams, Optional[LoRARequest]]]:
    """Create a list of test prompts with their sampling parameters.

    2 requests for base model, 3 requests for the LoRA. We define 3
    different LoRA adapters (using the same model for demo purposes).
    Since we also set `max_loras=3`, the expectation is that the requests
    with the second LoRA adapter will be ran after all requests with the
    first adapter have finished.
    """
    adapter_count = 0
    return_list: List[Tuple[str, SamplingParams, Optional[LoRARequest]]] = []
    for k in range (0, 2):
        adapter_count = 0
        for i in range (0, 13):
            return_list.extend([
            (
                llama3_instruct_prompt("What is the meaning of 'To be or not to be'"),
                SamplingParams(
                    temperature=0.0, max_tokens=128, stop=["<|eot_id|>"]
                ),
                None,
            ),
            (
                llama3_instruct_prompt(
                    "I want you to act as a SQL terminal in front of an example database, you need only to return the sql command to me.Below is an instruction that describes a task, Write a response that appropriately completes the request.\n\n##Instruction:\ncar_1 contains tables such as continents, countries, car_makers, model_list, car_names, cars_data. Table continents has columns such as ContId, Continent. ContId is the primary key.\nTable countries has columns such as CountryId, CountryName, Continent. CountryId is the primary key.\nTable car_makers has columns such as Id, Maker, FullName, Country. Id is the primary key.\nTable model_list has columns such as ModelId, Maker, Model. ModelId is the primary key.\nTable car_names has columns such as MakeId, Model, Make. MakeId is the primary key.\nTable cars_data has columns such as Id, MPG, Cylinders, Edispl, Horsepower, Weight, Accelerate, Year. Id is the primary key.\nThe Continent of countries is the foreign key of ContId of continents.\nThe Country of car_makers is the foreign key of CountryId of countries.\nThe Maker of model_list is the foreign key of Id of car_makers.\nThe Model of car_names is the foreign key of Model of model_list.\nThe Id of cars_data is the foreign key of MakeId of car_names.\n\n###Input:\nWhat are the different models for the cards produced after 1980?\n\n###Response:"
                ),
                SamplingParams(
                    temperature=0.0, max_tokens=128, stop=["<|eot_id|>"]
                ),
                LoRARequest("sql-lora", adapter_count + 1, sql_lora_path),
            ),
            (
                llama3_instruct_prompt("介绍一下机器学习"),
                SamplingParams(
                    temperature=0.0, max_tokens=128, stop=["<|eot_id|>"]
                ),
                LoRARequest("chinese-lora", adapter_count + 2, chinese_lora_path),
            ),
            (
                llama3_instruct_prompt("How to create a new docker container based on latest ubuntu?"),
                SamplingParams(
                    temperature=0.0, max_tokens=128, stop=["<|eot_id|>"]
                ),
                LoRARequest("docker-lora", adapter_count + 3, docker_lora_path),
            ),
            ])
            adapter_count += 3
    return return_list


def process_requests(
    engine: LLMEngine,
    test_prompts: List[Tuple[str, SamplingParams, Optional[LoRARequest]]],
):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    for prompt, sampling_params, lora_request in test_prompts:
        engine.add_request(str(request_id), prompt, sampling_params, lora_request=lora_request)
        request_id += 1

    request_outputs: List[RequestOutput] = []
    bench_global_vars.set_value("total_time", 0)
    bench_global_vars.set_value("lora_total_time", 0)
    bench_global_vars.set_value("lora_swap_time", 0)
    start_time = time.perf_counter_ns()
    while engine.has_unfinished_requests():
        request_outputs.extend(engine.step())
    end_time = time.perf_counter_ns()

    for request_output in request_outputs:
        if request_output.finished:
            print(request_output.outputs[0].text)
    
    print("Total time taken: ", (end_time - start_time) / 1e9)
    print("Total computing time: ", bench_global_vars.get_value("total_time") / 1e9)
    print("Total LoRA time: ", bench_global_vars.get_value("lora_total_time") / 1e9)
    print("Total LoRA count: ", bench_global_vars.get_value("lora_total_count"))
    print("Total LoRA swap time: ", bench_global_vars.get_value("lora_swap_time") / 1e9)


def initialize_engine() -> LLMEngine:
    """Initialize the LLMEngine."""
    # max_loras: controls the number of LoRAs that can be used in the same
    #   batch. Larger numbers will cause higher memory usage, as each LoRA
    #   slot requires its own preallocated tensor.
    # max_lora_rank: controls the maximum supported rank of all LoRAs. Larger
    #   numbers will cause higher memory usage. If you know that all LoRAs will
    #   use the same rank, it is recommended to set this as low as possible.
    # max_cpu_loras: controls the size of the CPU LoRA cache.
    engine_args = EngineArgs(
        model="/downloads/llama3-8b-awq",
        enable_lora=True,
        enforce_eager=True,
        max_loras=3,
        max_lora_rank=64,
        max_cpu_loras=3,
        use_page_cache=False,
        memory_pool_size=1024*3,
        memory_pool_page_size=64,
        max_num_seqs=256,
        quantization="awq",
        swap_space=16,
        dtype=torch.float16,
        lora_dtype=torch.float16,
        gpu_memory_utilization=0.9
    )
    return LLMEngine.from_engine_args(engine_args)

def initialize_page_cache_engine() -> LLMEngine:
    engine_args = EngineArgs(
        model="/downloads/llama3-8b-awq",
        enable_lora=True,
        enforce_eager=True,
        max_loras=16,
        max_lora_rank=64,
        max_cpu_loras=30,
        use_page_cache=True,
        memory_pool_size=1024,
        memory_pool_page_size=64,
        max_num_seqs=256,
        quantization="awq",
        swap_space=16,
        dtype=torch.float16,
        lora_dtype=torch.float16,
        gpu_memory_utilization=0.9
    )
    return LLMEngine.from_engine_args(engine_args)


def benchmark():
    """Main function that sets up and runs the prompt processing."""
    bench_global_vars._init()
    bench_global_vars.set_value("total_time", 0)
    bench_global_vars.set_value("lora_total_time", 0)
    bench_global_vars.set_value("lora_total_count", 0)
    bench_global_vars.set_value("lora_swap_time", 0)
    # engine = initialize_page_cache_engine()
    engine = initialize_engine()
    test_prompts = create_test_prompts()
    process_requests(engine, test_prompts)


if __name__ == "__main__":
    benchmark()
