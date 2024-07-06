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
DTYPE = torch.float16
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
    vision_language_config=None, 
    parallel_config=None,
    scheduler_config = None,
    cache_config = None)

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

prompts = [
    "The meaning of life is",
    "The social media platform is",
    "The best way to cook a turkey is",
    "The best way to learn a new language is",
    "The best way to learn a new language is"
]

bs = len(prompts)
input_ids = tokenizer(prompts)['input_ids']
position_ids = [list(range(len(_))) for _ in input_ids]
seq_lens = [len(_) for _ in input_ids]
past_seq_lens = [0] * bs 

kv_caches = [[(torch.zeros((MAX_SEQ_LEN, HIDDEN_SIZE), dtype = DTYPE, device = DEVICE),
              torch.zeros((MAX_SEQ_LEN, HIDDEN_SIZE), dtype = DTYPE, device = DEVICE)) for _ in range(bs)] 
                for _ in range(n_layer)]

input_ids = torch.tensor(sum(input_ids, []), dtype = torch.int32, device = DEVICE)
position_ids = torch.tensor(sum(position_ids, []), dtype = torch.int32, device = DEVICE)

# ensor([   2,  133, 3099,    9,  301,   16], device='cuda:0'
print('input_ids:', input_ids)

result = model(
    input_ids, 
    position_ids,
    kv_caches,
    SeqAttnMetadata(seq_lens, past_seq_lens)
)

# print('result', result)
# exit(0)

# print('kv_caches[0]', kv_caches[0][0][0])

accumulated_length = [0] * bs
for i in range(bs):
    accumulated_length[i] = accumulated_length[i-1] + seq_lens[i] if i > 0 else seq_lens[i]
accumulated_length = [_ - 1 for _ in accumulated_length]
# print('past_seq_lens', past_seq_lens)
# print('seq_lens', seq_lens)
# print('result', result.size())
# print('accumulated_length', accumulated_length)
logits = result[accumulated_length] @ model.lm_head_weight.T
input_ids = torch.argmax(logits, dim = -1)
past_seq_lens = seq_lens
seq_lens = [1] * bs

outputs = ['' for _ in range(bs)]
for i in range(100):
    decoded = tokenizer.batch_decode(input_ids)
    for _ in range(bs):
        outputs[_] += decoded[_]
    position_ids = torch.tensor(past_seq_lens, dtype = torch.int32, device = DEVICE)
    # print('position_ids', position_ids)
    # print('input_ids', input_ids)
    # print('seq_lens', seq_lens)
    # print('past_seq_lens', past_seq_lens)
    result = model(
        input_ids, 
        position_ids,
        kv_caches,
        SeqAttnMetadata(seq_lens, past_seq_lens)
    )
    logits = result @ model.lm_head_weight.T
    input_ids = torch.argmax(logits, dim = -1)
    past_seq_lens = [_ + 1 for _ in past_seq_lens]
for prompt, generated in zip(prompts, outputs):
    print(f"Prompt: {prompt!r}, Generated text: {generated!r}")