import re 

with open("the-verdict.txt", "r") as f:
    raw_text = f.read()
print(f"Total number of characters:{len(raw_text)}")

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item for item in preprocessed if item.strip()]


all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
print("Number of words in initial text:",len(all_tokens))

vocab = {}
index = 0

for token in all_tokens:
    vocab[token] = index
    index += 1

print(vocab)



class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids


    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
    

tokenizer= SimpleTokenizerV1(vocab)
text = "Hello, do you like tea?"

ids = tokenizer.encode(text)
print(ids)

words = tokenizer.decode(ids)
print(words)
