from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import numpy as np
import json
import pickle
import os
import time
import gc


print("""            
      
      

            *** NOTICE:The pkl path (variable 'pkl_path') in the code is default to the root directory ***
    `*** the pkl files are huge (estimated 240GB to 480GB in total) ***
    *** Please make sure you have enough space in the root directory, or you can change the path to other directory ***
      

      
      """)

system_prompt = {
'basic' :"""请仔细阅读以下多条文本,然后回答问题：基于这些文本，这个用户是否存在抑郁情绪？用“是”或者“否”回答
***重要***
你必须用以下格式回答问题
答案：“是”或“否”
""",
'OR': """请仔细阅读以下多条文本，关注用户的情绪表达、对未来的态度、社交参与度、兴趣水平以及任何关于身体症状或睡眠/食欲变化、自我价值、个人感受的表述。回答问题：基于这些文本，这个用户是否存在抑郁情绪？用“是”或者“否”回答，再给出你一步一步的分析过程。
***重要***
你必须用以下格式回答问题
答案：“是”或“否”
分析：你一步一步的分析过程，解释你为什么给出这个答案。""",
'OR2': """请仔细阅读以下多条文本然后回答问题：基于这些文本，这个用户是否存在抑郁情绪？用“是”或者“否”回答，再给出你一步一步的分析过程但是不要直接引用原文。
***重要***
你必须用以下格式回答问题
答案：“是”或“否”
分析：你一步一步的分析过程，解释你为什么给出这个答案。""",
'OR3': """请仔细阅读以下多条文本然后回答问题：基于这些文本，这个用户是否存在抑郁情绪？用“是”或者“否”回答，再给出你一步一步的分析过程。
""",
'RET':"""请逐条阅读以下用户的文本，关注用户的情绪表达、对未来的态度、社交参与度、兴趣水平以及任何关于身体症状或睡眠/食欲变化、自我价值、个人感受的表述。回答问题：基于这些文本，这个用户是否存在抑郁情绪？用“是”或者“否”回答，再给出你一步一步的分析过程，指出哪条或哪几条文本让你得出这个回答。
***重要***
如果用户在文本中明确提到自己得到了抑郁的诊断，请直接判断为“是”，并在分析中指出相关文本。
你必须用以下格式回答问题
答案：“是”或“否”
分析：你一步一步的分析过程，指出哪条或哪几条文本让你得出这个回答。""",
'RET-REV':"""请逐条阅读以下用户的文本，关注用户的情绪表达、对未来的态度、社交参与度、兴趣水平以及任何关于身体症状或睡眠/食欲变化、自我价值、个人感受的表述。先分析这些文本并给出你一步一步的分析过程，然后回答这个用户是否存在抑郁情绪？，。
***重要***
如果用户在文本中明确提到自己得到了抑郁的诊断，请直接判断为“是”，并在分析中指出相关文本。
你必须用以下格式回答问题
分析：你一步一步的分析过程，指出哪条或哪几条文本让你得出这个回答。
答案：“是”或“否”""",
'OR2-REV': """请仔细阅读以下多条文本,通过这些文本一步一步的分析这个用户是否存在抑郁情绪，给出你的分析过程，然后回答这个用户是否存在抑郁情绪？
***重要***
你必须用以下格式回答问题
分析：你一步一步的分析过程，解释你为什么给出这个答案。
答案：“是”或“否”
""",
'OR3-REV': """请仔细阅读以下多条文本,通过这些文本一步一步的分析这个用户是否存在抑郁情绪，给出你的分析过程，然后用“是”或者“否”回答这个用户是否存在抑郁情绪。
""",}

def modify_tweet (tweet,mode = 1):
    input_text_list = tweet.split('\n')
    if mode == 1:
        input_text_list_mod= [f'文本{i+1}:{text}' for i,text in enumerate(input_text_list)]
        tweet = '\n'.join(input_text_list_mod)
    elif mode == 2:
        input_text_list_mod = [f'<文本{i+1}>{text}</文本{i+1}> ' for i,text in enumerate(input_text_list)]
        tweet = '\n'.join(input_text_list_mod)
    elif mode == 0:
        tweet = tweet
    return tweet
#预处理需要的数据
def find_start_end_index(start_token = '<|im_start|>',end_token ='<|im_end|>',drift = 2):
    input_start = 0
    input_end = 0
    output_start = 0
    output_end = 0
    input_count = 0
    output_count = 0
    i = 0
    for token in generated_ids[0][0]:
        if tokenizer.decode(token) == start_token:
            if input_count == 0:
                input_start = i+drift
                input_count += 1
            else:
                output_start = i+drift
        elif tokenizer.decode(token) == end_token:
            if output_count == 0:
                input_end = i
                output_count += 1
            else:
                output_end = i
        i += 1
    print('input_start:',input_start,'input_end:',input_end)
    print('output_start:',output_start,'output_end:',output_end)
    assert input_start < input_end
    #assert output_start < output_end
    return input_start,input_end,output_start,output_end
def input_reviced_attention(start,end,mode ='max'):
    s = start
    t = end
    if mode == 'max':
        attention_input = [
            torch.max(hd[:, :, s:t, s:t].cpu().squeeze(0).to(torch.float32), dim=0)[0]
            for hd in generated_ids['attentions'][0]
        ]
    elif mode == 'mean':
        attention_input = [
            torch.mean(hd[:, :, s:t, s:t].cpu().squeeze(0).to(torch.float32), dim=0)
            for hd in generated_ids['attentions'][0]
        ]
    #print(len(attention_input))
    print(attention_input[0].shape)
    reviced_attention = []
    #attender_attention = []
    index_tokens = [ tokenizer.decode(t,skip_special_tokens=False) for t in generated_ids[0][0][s:t]]
    seq_len = len(index_tokens)
    for layer in attention_input:
        total_attention_as_object = torch.sum(layer, axis=0).tolist()
        total_attention_as_object = [att/seq_len for att in total_attention_as_object]
        #total_attention_as_attender = torch.sum(layer, axis=1).tolist()
        #total_attention_as_attender = [att/seq_len for att in total_attention_as_attender]
        reviced_attention.append(total_attention_as_object)
        #attender_attention.append(total_attention_as_attender)
    return reviced_attention
def input_reviced_attention_rollout(reviced_attention,output_mode = 'lastlayer'):
    reviced_attention_tensor = [np.array(layer) for layer in reviced_attention]

    bias_vector = np.ones(reviced_attention_tensor[0].shape[0])
    bias_vector = bias_vector[None, :]
    reviced_attention_tensor = np.array(reviced_attention_tensor) + bias_vector

    reviced_attention_tensor = rmsnorm(torch.tensor(reviced_attention_tensor)).numpy()   
    layers = reviced_attention_tensor.shape[0]
    joint_attention = np.zeros(reviced_attention_tensor.shape)
    joint_attention[0] = reviced_attention_tensor[0]
    for i in np.arange(1, layers):
        joint_attention[i] = reviced_attention_tensor[i] * joint_attention[i-1]  #逐元素相乘,因为原文中的公式是用于计算注意力方阵的[seq_len,seq_len]，但是这里用于计算注意力向量[1,seq_len]，所以需要逐元素相乘
    if output_mode == 'lastlayer':
        return joint_attention[-1]
    elif output_mode == 'maxpooling':
        return np.max(joint_attention, axis=0)
    elif output_mode == 'meanpooling':
        return np.mean(joint_attention, axis=0)
    else:
        raise ValueError("Invalid output mode. Please select from 'lastlayer', 'maxpooling', or 'meanpooling'.")

def rmsnorm(x, eps=1e-6):
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return x / rms

with open ('negative_samples.json','r',encoding='utf-8') as f:
    negative_samples = json.load(f)
with open ('positive_samples.json','r',encoding='utf-8') as f:
    positive_samples = json.load(f)


workpath = os.getcwd()
# Please modify the the path accordinglly
result_path = os.path.join(workpath,'result')
pkl_path = os.path.join(workpath,'pkl')
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(pkl_path):
    os.makedirs(pkl_path)
model_id = '/root/autodl-tmp/Llama3-8B-Chinese-Chat'
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager" ,
    offload_buffers=False,
    #max_memory=max_memory
).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.generation_config.temperature=None
model.generation_config.top_p=None
model.generation_config.top_k=None


wait_time = 1  # Time to wait between each sample

qwen_generate_params = {
    "max_new_tokens": 350,
    "output_attentions": True,
    #"output_hidden_states": True,
    "return_dict_in_generate": True,
    "do_sample": False,
    #"temperature": 1.2,
    #"top_p": 0.9,
}

OOM = []
device = 'cuda' 
for samples,sample_type in zip([positive_samples,negative_samples],['positive','negative']):
    OOM_path = os.path.join(result_path,f'{sample_type}_OOM.json')
    for prompt_tag in ['OR2']:
        mode = 0 if 'OR' in prompt_tag else 1
        c = 1
        output_reuslt_path = os.path.join(
        result_path,
        f"{sample_type}_result_{prompt_tag}_tmp{qwen_generate_params['temperature']}_topp{qwen_generate_params['top_p']}.json"
            ) if qwen_generate_params['do_sample'] else os.path.join(
                result_path,
                f"{sample_type}_result_{prompt_tag}.json"
            )
        try:
            already_done = list(set([key for line in open(output_reuslt_path, 'r', encoding='utf-8') for key in json.loads(line.strip()).keys()]))
        except FileNotFoundError :
            already_done = []
        print(f'Now running {sample_type} samples with prompt {prompt_tag}')
        sample_list = list(samples.keys())
        sample_list = [sample for sample in sample_list if sample not in already_done]
        for user in sample_list:
            output_pkl_path = os.path.join(
            pkl_path,
            f"{sample_type}_{prompt_tag}_tmp{qwen_generate_params['temperature']}_topp{qwen_generate_params['top_p']}_final_{user}.pkl"
                ) if qwen_generate_params['do_sample'] else os.path.join(
                    pkl_path,
                    f"{sample_type}_{prompt_tag}_final_{user}.pkl"
                )
            try:
                tweet  =  modify_tweet(tweet = samples[user][0],mode = mode)
                result = {}
                response = None

                if 'qwen' in model_id.lower():
                    generate_params = qwen_generate_params
                    prompt = system_prompt[prompt_tag] + '\n' + tweet
                    messages = [{"role": "system", "content": prompt}]
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    with torch.no_grad():
                        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                        generated_ids = model.generate(model_inputs.input_ids, **generate_params)
                    response_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids[0])]
                    response = tokenizer.batch_decode(response_ids, skip_special_tokens=False)[0]
                #Not tested yet
                elif 'llama3' in model_id.lower():
                    prompt = system_prompt[prompt_tag] + '\n' + tweet
                    messages = [{"role": "system", "content": prompt}]
                    model_inputs = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    ).to(model.device)

                    terminators = [
                        tokenizer.eos_token_id,
                        tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]
                    generated_ids = model.generate(
                        model_inputs,
                        eos_token_id=terminators,
                        pad_token_id=tokenizer.eos_token_id,
                        **qwen_generate_params
                    )
                    response_ids = generated_ids[0][0][model_inputs.shape[-1]:]
                    response = tokenizer.decode(response_ids, skip_special_tokens=False)
                else:
                    raise ValueError('Model not supported')
                print(f'Now running {sample_type} samples with prompt {prompt_tag}')
                print(f'{c}/{len(sample_list)}: {response[:20]}')#)
                print(f'Dumping Results and pkl files to the hard drive')
                result[user] = response
                with open(output_reuslt_path, 'a', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False)
                    f.write('\n')
            except torch.cuda.OutOfMemoryError as e:
                print(f'Out of Memory: {user}')
                OOM.append(user)
                with open(OOM_path, 'a', encoding='utf-8') as f:
                    json.dump(OOM, f, ensure_ascii=False)
                    f.write('\n')
                #os.system("/usr/bin/shutdown")
                try:
                    if 'joint_attentions' in locals():
                        del joint_attentions
                    if 'self_joint_attention' in locals():
                        del self_joint_attention
                    if 'last_tokens_attention' in locals():
                        del last_tokens_attention
                    if 'reviced_attention' in locals():
                        del reviced_attention
                    if 'recived_pairs' in locals():
                        del recived_pairs
                    if 'index_tokens' in locals():
                        del index_tokens
                    if 'seq_len' in locals():
                        del seq_len
                    if 'attentions_step1' in locals():
                        del attentions_step1
                    if 'last_token_attention_list' in locals():
                        del last_token_attention_list
                    if 'converted_attentions' in locals():
                        del converted_attentions
                    if 'result' in locals():
                        del result
                    if 'final_result' in locals():
                        del final_result
                    if 'response' in locals():
                        del response
                    if 'response_ids' in locals():
                        del response_ids
                    if 'model_inputs' in locals():
                        del model_inputs
                    if 'generated_ids' in locals():
                        del generated_ids
                    if 'prompt' in locals():
                        del prompt
                    if 'messages' in locals():
                        del messages
                    if  'res_att_mat' in locals():
                        del res_att_mat
                    if 'bias_vector' in locals():
                        del bias_vector
                    if 'joint_att' in locals():
                        del joint_att
                    if 'layers' in locals():
                        del layers
                    if 'eps' in locals():
                        del eps
                    if 'all_attentions' in locals():
                        del all_attentions
                    if 'model' in locals():
                        del model
                    if 'tokenizer' in locals():
                        del tokenizer
                    torch.cuda.empty_cache()
                    gc.collect()
                    time.sleep(2)
                    print('Memory Cleared for OOM')     
                except Exception as memory_clear_exception:
                    pass
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        attn_implementation="eager",
                        offload_buffers=False,
                    )
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    model.generation_config.temperature=None
                    model.generation_config.top_p=None
                    model.generation_config.top_k=None
                    print('Model Reloaded')
                    continue
                except Exception as reload_exception:
                    print(f'Failed to reload model: {reload_exception}')
            input_start, input_end, output_start, output_end = find_start_end_index(start_token='<|end_header_id|>',end_token='<|eot_id|>',drift=2)
            s = input_start
            t = input_end
            attention_input = [torch.max(hd[:, :, s:t, s:t].cpu().squeeze(0).to(torch.float32), dim=0)[0]
                                for hd in generated_ids['attentions'][0]]
            reviced_attention = []
            recived_pairs = []
            index_tokens = [ tokenizer.decode(t,skip_special_tokens=False) for t in generated_ids[0][0][s:t]]
            seq_len = len(index_tokens)
            for layer in attention_input:
                
                total_attention_as_object = torch.sum(layer, axis=0).tolist()
                total_attention_as_object = [att/seq_len for att in total_attention_as_object]
                reviced_attention.append(total_attention_as_object)
            reviced_attention_tensor = [np.array(layer) for layer in reviced_attention]
            bias_vector = np.ones(reviced_attention_tensor[0].shape[0])
            bias_vector = bias_vector[None, :]
            reviced_attention_tensor = np.array(reviced_attention_tensor) + bias_vector
            reviced_attention_tensor = rmsnorm(torch.tensor(reviced_attention_tensor)).numpy()   
            layers = reviced_attention_tensor.shape[0]
            joint_attention = np.zeros(reviced_attention_tensor.shape)
            joint_attention[0] = reviced_attention_tensor[0]
            for i in np.arange(1, layers):
                joint_attention[i] = reviced_attention_tensor[i] * joint_attention[i-1] 
            self_joint_attention = joint_attention

            
            attentions_step1 = generated_ids['attentions'][0]
            
            last_token_attention_list = []
            assert generated_ids[0][0][input_end:input_end+1] == tokenizer.eos_token_id
            
            for layer_attention in attentions_step1:
                
                last_token_attention = layer_attention[:, :, input_end:input_end+1:, :]
                last_token_attention_list.append(last_token_attention)
            
            attentions_list = list(generated_ids['attentions'])
           
            attentions_list[0] = tuple(last_token_attention_list)
            
            generated_ids['attentions'] = tuple(attentions_list)

            converted_attentions = []

           
            num_layers = generated_ids['attentions'][0][0].shape[1]  # 28
            num_heads = generated_ids['attentions'][0][0].shape[1] # 28
            num_steps = len(generated_ids['attentions'])  

            
            #last_tokens_attention = torch.zeros((num_layers, num_heads, num_steps))
            last_tokens_attention = []
            assert generated_ids[0][0][input_end:input_end+1] == tokenizer.eos_token_id
            

            for i in range(num_steps):
                step_attentions = torch.cat([layer for layer in generated_ids['attentions'][i]], dim=0)
                last_token_attention = step_attentions[:, :, -1, -2]
                if step_attentions.shape[-1] > input_start:
                    step_attentions = step_attentions[:, :, :, input_start:input_end]
                
            
                converted_attentions.append(step_attentions.detach().clone().cpu().to(torch.float32).numpy())
                
                
                if i > 1: #从第3个生成步骤开始
                    
                    last_tokens_attention.append(last_token_attention.detach().clone().cpu().to(torch.float32).numpy())


            
            last_tokens_attention.extend([np.ones((num_layers,num_layers)),np.ones((num_layers,num_layers))])
            last_tokens_attention = np.array(last_tokens_attention)
            last_tokens_attention = last_tokens_attention.transpose((1,2,0)) #将生成步数放在最后一个维度
            eps = 1e-6
            all_attentions =  [(t,a) for t,a in zip(response_ids[0].tolist(), converted_attentions)]
            joint_attentions =[]
            for pair in all_attentions:
                token = pair[0]
                res_att_mat = pair[1].max(axis=1) 
                bias_vector = np.ones(pair[1].shape[-1])
                bias_vector = bias_vector[None, None, :]
                res_att_mat = res_att_mat + bias_vector #对[layers,1,seq_len]
                res_att_mat = rmsnorm(torch.tensor(res_att_mat), eps=eps).numpy() 
                joint_att = np.zeros(res_att_mat.shape)
                layers = joint_att.shape[0]
                joint_att[0] = res_att_mat[0]
                for i in np.arange(1, layers):
                    joint_att[i] = res_att_mat[i] * joint_att[i-1]
                joint_attentions.append((token,joint_att))
            final_result = {'response':response,
                            'response_ids':response_ids,
                            'model_inputs':model_inputs,
                            'joint_attentions':joint_attentions,
                            'self_joint_attention':self_joint_attention,
                            'last_tokens_attention':last_tokens_attention,
                            }

            with open(output_pkl_path, 'wb') as f:
                pickle.dump(final_result, f)
            if 'joint_attentions' in locals():
                del joint_attentions
            if 'self_joint_attention' in locals():
                del self_joint_attention
            if 'last_tokens_attention' in locals():
                del last_tokens_attention
            if 'reviced_attention' in locals():
                del reviced_attention
            if 'recived_pairs' in locals():
                del recived_pairs
            if 'index_tokens' in locals():
                del index_tokens
            if 'seq_len' in locals():
                del seq_len
            if 'attentions_step1' in locals():
                del attentions_step1
            if 'last_token_attention_list' in locals():
                del last_token_attention_list
            if 'converted_attentions' in locals():
                del converted_attentions
            if 'result' in locals():
                del result
            if 'final_result' in locals():
                del final_result
            if 'response' in locals():
                del response
            if 'response_ids' in locals():
                del response_ids
            if 'model_inputs' in locals():
                del model_inputs
            if 'generated_ids' in locals():
                del generated_ids
            if 'prompt' in locals():
                del prompt
            if 'messages' in locals():
                del messages
            if  'res_att_mat' in locals():
                del res_att_mat
            if 'bias_vector' in locals():
                del bias_vector
            if 'joint_att' in locals():
                del joint_att
            if 'layers' in locals():
                del layers
            if 'eps' in locals():
                del eps
            if 'all_attentions' in locals():
                del all_attentions
            del model
            del tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="eager" ,
                offload_buffers=False,
                #max_memory=max_memory
            ).to('cuda')
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model.generation_config.temperature=None
            model.generation_config.top_p=None
            model.generation_config.top_k=None
            print(f'Memory Cleared')
            print(f'Waiting for {wait_time} seconds')
            time.sleep(wait_time)
            c += 1
time.sleep(60) # Wait for the uploader to finish
os.system("/usr/bin/shutdown")
