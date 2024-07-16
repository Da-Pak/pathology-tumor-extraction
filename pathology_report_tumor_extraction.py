import pandas as pd
import numpy as np
import torch
from os.path import expanduser
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.chat_models import Llama2Chat
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.prompts import FewShotChatMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_community.llms import LlamaCpp
from datetime import datetime
from time import time
from wurlitzer import pipes

# There are 1000 microscopic descriptions for 'breast' and 1000 for 'lung', making a total of 2000 descriptions.
# Load the pathology records
df = pd.read_csv('pathology_records.csv', encoding='cp949')

# Function to run the model
def run(model_name, lung_breast, shot_value):
    # Example data for few-shot learning
    examples = {
        'breast': [
            {"input": "...", "output": "{...}"},
            {"input": "...", "output": "{...}"},
            {"input": "...", "output": "{...}"},
            {"input": "...", "output": "{...}"},
            {"input": "...", "output": "{...}"},# Include 5 examples here
        ],
        'lung': [
            {"input": "...", "output": "{...}"},
            {"input": "...", "output": "{...}"},
            {"input": "...", "output": "{...}"},
            {"input": "...", "output": "{...}"},
            {"input": "...", "output": "{...}"},# Include 5 examples here
        ]
    }
        
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples[lung_breast]
    )
    
    # Constructing the template messages
    template_messages = [
        SystemMessage(content="Please extract 'tumor size' and 'tumor site' from pathology reports and present the information in JSON format, Let's think step by step"),
        few_shot_prompt if shot_value == 5 else None,
        HumanMessagePromptTemplate.from_template("{text}")
    ]

    prompt_template = ChatPromptTemplate.from_messages(template_messages)
   
   # Load the model
    model_path = expanduser(model_name)
    llm = LlamaCpp(
        model_path=model_path,
        streaming=False,
        n_gpu_layers=-1,
        n_ctx = 4096,
        stop = ['</s>','[INST]','[INST', '[INSTS]', '[INST:]']
    )
    model = Llama2Chat(llm=llm)
    chain = LLMChain(llm=model, prompt=prompt_template)

    # Filter the DataFrame based on sample type
    df2 = df[df['SampleName'] == lung_breast]
    df2['micro'] = df2['micro'].str.lower()

    start = time()

    response_list = []
    err_list = []
    for N, record in enumerate(df2['micro'][:1000]):
        print(model_name, shot_value, N)
        # Using pipes() to capture stderr for LlamaCpp token generation speed
        with pipes() as (out, err):
            if pd.isna(record):
                response_list.append(np.nan)
                continue
            response = chain.invoke(input=record)
            response_list.append(response['text'])
            print('llm result :',response['text'])
        err_list.append(err.getvalue()) # Capture LlamaCpp token generation speed information     
        
    end = time()
    spend_time = np.round((end - start)/60,0)
        
    df2['generated'] = response_list
    df2['err'] = err_list

    from datetime import datetime
    now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    head_word = model_name.split('./models/llama-2-')[1].split('-chat.Q4_K_M.gguf')[0]
    save_file_name = f'{head_word}_result_forth_test_size_and_site_{shot_value}shot_test_{lung_breast}_{now}_{spend_time}.csv'
    df2.to_csv(save_file_name)
    
    del model
    torch.cuda.empty_cache()
    return save_file_name


# Main execution loop
sample_types = ['lung', 'breast']
shot_value_list = [5, 0]
model_name_list = ["./models/llama-2-7b-chat.Q4_K_M.gguf", "./models/llama-2-13b-chat.Q4_K_M.gguf", "./models/llama-2-70b-chat.Q4_K_M.gguf"]

completed_file_list = [run(model_name, sample_type, shot_value) for sample_type in sample_types for shot_value in shot_value_list for model_name in model_name_list]
print(completed_file_list)