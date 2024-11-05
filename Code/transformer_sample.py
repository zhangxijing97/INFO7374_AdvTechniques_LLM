from transformers import pipeline

unmasker = pipeline('fill-mask', model='bert-base-uncased')
print(unmasker("The man worked as a [MASK]."))
print(unmasker("The woman worked as a [MASK]."))


analyze_sentiment = pipeline('text-classification',model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
print(analyze_sentiment("The movie was bad"))


summarize = pipeline('summarization', model='facebook/bart-large-xsum')
text = summarize("""Machine has displaced human labour in today's factories and workshops.
                           Machines have mostly taken over tasks that were once performed by hand.
                           Machines has mostly supplanted human workers in telemarketing and customer support""",
                 min_length=5, max_length=40, do_sample=False)
print(text[0]['summary_text'])


translate = pipeline('translation_en_to_fr', model='google-t5/t5-base')
text= translate(" BART is able to handle a range of NLP challenges effectively")
print(text[0]['translation_text'])

translate = pipeline('translation', src_lang='fr',tgt_lang='en',model='facebook/m2m100_418M')
text= translate(text[0]['translation_text'])
print(text[0]['translation_text'])


# from openai import OpenAI

# key= '' #get a key from: https://platform.openai.com/api-keys
# query = 'complete the following scentence with one word. just give me three names of most probable words: this is a nice'
# client = OpenAI(api_key=key)
# completion = client.chat.completions.create(
#             model='gpt-4o',
#             messages=[{"role": "user", "content": query}],
#             logprobs=True,
#             temperature= 1,
#         )
# print(completion.choices[0].message.content)
# print(completion.choices[0].logprobs.content)

