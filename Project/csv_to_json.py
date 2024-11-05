import pandas as pd
import json

def save_to_jsonl(data, output_file_path, query_title = 'user', response_title = 'assistant'):
    jsonl_data = []
    for index, row in data.iterrows():
        jsonl_data.append({
            "messages": [
                {"role": query_title, "content": row['Query']},
                {"role": response_title, "content": f"\"{row['Response']}\""}
            ]
        })

    # Save to JSONL format
    with open(output_file_path, 'w') as f:
        for item in jsonl_data:
            f.write(json.dumps(item) + '\n')



data = pd.read_csv('FineTuning_Data.csv')

#GPT
save_to_jsonl(data , 'FineTuning_Data_GPT.jsonl')

#Cohere
#save_to_jsonl(data , 'FineTuning_Data_Cohere.jsonl',query_title='User',response_title='Chatbot')

