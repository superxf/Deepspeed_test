import modelingpreln_layerdrop as modeling
import torch
import yaml
import  numpy as np

device = torch.device("cuda", 0)
with open('config.yaml','r',encoding='utf8')as f:
    args=yaml.load(f,Loader=yaml.FullLoader)

class Args(object): 
    def __init__(self, args):
        self.deepspeed_sparse_attention = args[ 'deepspeed_sparse_attention' ]
        self.sparse_attention =  args['sparse_attention' ]
        self.mode = args['mode']
        self.deepspeed_transformer_kernel = args['deepspeed_transformer_kernel']
        self.train_micro_batch_size_per_gpu = args['train_micro_batch_size_per_gpu']
        self.max_seq_length = args['max_seq_length']
        self.fp16_enabled  =  args['fp16_enabled']
args =  Args(args)

input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]]).to(device)
input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]]).to(device)
token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]]).to(device)

config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
    num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

model = modeling.BertModel(config=config, args=args).to(torch.half).to(device)
all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)