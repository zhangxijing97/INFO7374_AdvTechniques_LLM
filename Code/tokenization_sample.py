from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import CharBPETokenizer

# Instantiate tokenizer
tokenizer = CharBPETokenizer()

tokenizer.train([ "./sample.txt"],min_frequency=2, vocab_size=30)

print(tokenizer.get_vocab())

output = tokenizer.encode("highest")
print(output.tokens)

output = tokenizer.encode("newer")
print(output.tokens)
print(output.ids)


output = tokenizer.encode("higher is better")
print(output.tokens)



