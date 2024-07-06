import os 

os.environ['VLLM_ATTENTION_BACKEND'] = 'SEQ_ATTN'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
from vllm.model_executor.models.opt import OPTForCausalLM
from transformers import AutoTokenizer 

from transformers import AutoConfig
import torch 
from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
from vllm.attention.backends.seq_attn import SeqAttnMetadata
from vllm.model_executor.model_loader import get_model_loader
from vllm.config import LoadConfig, ModelConfig, DeviceConfig, ParallelConfig, CacheConfig
init_distributed_environment()
initialize_model_parallel()

config = AutoConfig.from_pretrained("facebook/opt-125m")

n_layer = config.num_hidden_layers
MAX_SEQ_LEN = 1024
HIDDEN_SIZE = config.hidden_size
DTYPE = torch.float32
DEVICE = torch.device('cuda')

model_loader = get_model_loader(LoadConfig())
model_config = ModelConfig(
    model = "facebook/opt-125m",
    tokenizer = "facebook/opt-125m",
    tokenizer_mode = 'auto',
    trust_remote_code=True, 
    dtype = DTYPE,
    seed = 0
)
device_config = DeviceConfig('cuda')
para_config = ParallelConfig(
    pipeline_parallel_size=1,
    tensor_parallel_size=1,
)

model = model_loader.load_model(
    model_config = model_config, 
    device_config = device_config,
    lora_config = None,
    multimodal_config=None, 
    parallel_config=None,
    scheduler_config = None,
    cache_config = None)

from transformers import AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

prompts = [
    "The meaning of life is",
    # "The social media platform is",
    # "The best way to cook a turkey is",
    # "The best way to learn a new language is",
    # "The best way to learn a new language is"
]

bs = len(prompts)
input_ids = tokenizer(prompts)['input_ids']
position_ids = [list(range(len(_))) for _ in input_ids]
seq_lens = [len(_) for _ in input_ids]
past_seq_lens = [0] * bs 

kv_caches = [[(torch.rand((MAX_SEQ_LEN, HIDDEN_SIZE), dtype = DTYPE, device = DEVICE),
              torch.rand((MAX_SEQ_LEN, HIDDEN_SIZE), dtype = DTYPE, device = DEVICE)) for _ in range(bs)] 
                for _ in range(n_layer)]

input_ids = torch.tensor(sum(input_ids, []), dtype = torch.int32, device = DEVICE)
position_ids = torch.tensor(sum(position_ids, []), dtype = torch.int32, device = DEVICE)

result = model(
    input_ids, 
    position_ids,
    kv_caches,
    SeqAttnMetadata(seq_lens, past_seq_lens)
)

print('res:', result)