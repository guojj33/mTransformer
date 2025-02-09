import regex as re
import json

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    # [33, 126], [161, 172], [174, 255] # [0,31]是控制字符，32是空格，127是删除，[128,159]是latin-1的控制字符，160是不换行空格，173软连体字符
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n) # 32是空格字符，就变成chr(256+32)='Ġ'，占两个字节
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))    # int -> unicode char

def get_vocab_pairs(vocab):
    pairs = {}
    for word, freq in vocab.items():
        symbols = word.split(' ')   # word 'hel lo'
        for i in range(len(symbols)-1):
            pair = (symbols[i], symbols[i+1])
            if pair in pairs:
                pairs[pair] += freq
            else:
                pairs[pair] = freq
    return pairs

def merge_bigram_in_word(word, bigram): 
    '''
    input:
        word ('h', 'e', 'l', 'l', 'o')
        bigram ('h', 'e')
    output:
        new_word ('he', 'l', 'l', 'o')
    '''
    first, second = bigram
    new_word = []
    # merge bigram in word to get new_word
    i = 0
    while i < len(word):
        try:
            j = word.index(first, i) # find index of <first> starting from pos <i>
        except ValueError:
            new_word.extend(word[i:])   # not found, copy all elements after <i>
            break
        else:
            new_word.extend(word[i:j])  # found, copy elements before pos <i> to <new_word>
            i = j   # index of <first>

        if word[i] == first and i < len(word) - 1 and word[i+1] == second:
            new_word.append(first + second) # merge
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return new_word

def merge_vocab(vocab, pair):
    new_vocab = {}
    bigram = pair
    for word in vocab:  # word 'h e l l o'
        new_word = ' '.join(merge_bigram_in_word(tuple(word.split(' ')), bigram))
        new_vocab[new_word] = vocab[word]
    return new_vocab

def get_pairs(word):   
    '''
    input:
        word ('hel', 'lo')
    output:
        set of pairs in word
    '''
    pairs = set()
    prev_str = word[0]
    for str in word[1:]:
        pairs.add((prev_str, str))
        prev_str = str
    return pairs

class BPETokenizer:
    def __init__(self):

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.vocabulary = self.init_vocabulary()
        self.vocabulary_reverse = None
        self.merge_ops = self.init_merge_operations()
        self._bpe_ranks = None
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        self.vocab = {} # for training
        self.cache = {} # token to bpe tokens: 'hello' -> 'hel lo'

    def vocabulary_size(self):
        return len(self.vocabulary)

    def bpe_ranks(self, merge_op):
        if self._bpe_ranks == None or not (len(self._bpe_ranks) == len(self.merge_ops)):
            self._bpe_ranks = dict(zip(self.merge_ops, range(len(self.merge_ops))))
        if not merge_op in self._bpe_ranks:
            return float("inf")
        return self._bpe_ranks[merge_op]

    def init_vocabulary(self):
        '''
        init vocabulary with characters        
        '''
        vocabulary = {}
        n = 0
        for k, v in self.byte_encoder.items():
            vocabulary[v] = n
            n += 1
        return vocabulary
    
    def append_to_vocabulary(self, token):
        '''
        append new token to the end of the vocabulary
        '''
        if not token in self.vocabulary:
            self.vocabulary[token] = len(self.vocabulary)
    
    def init_merge_operations(self):
        '''
        init merge operations
        '''
        merge_ops = []
        return merge_ops
    
    def load(self, dir, load_state=False):
        # load vocabulary
        with open('{}/vocab.json'.format(dir), 'r') as f:
            self.vocabulary = json.load(f)
        # load merge operations
        self.merge_ops = self.init_merge_operations()
        with open('{}/merges.txt'.format(dir), 'r') as f:
            lines = f.readlines()
            for l in lines:
                self.merge_ops.append(tuple(l[:-1].split(' ')))
        print('load merge operation count: {}'.format(len(self.merge_ops)))
        print('load vocabulary size: {}'.format(len(self.vocabulary)))

        if load_state:
            with open('{}/state.json'.format(dir), 'r') as f:
                self.vocab = json.load(f)
            print('load state')

    def save(self, dir, save_state=False):
        # save vocabulary
        with open('{}/vocab.json'.format(dir), 'w') as f:
            json.dump(self.vocabulary, f)
        # save merge operations
        with open('{}/merges.txt'.format(dir), 'w') as f:
            for op in self.merge_ops:
                f.write('{} {}\n'.format(op[0], op[1]))
        
        if save_state:
            with open('{}/state.json'.format(dir), 'w') as f:
                json.dump(self.vocab, f)
            print('save state')


    def train(self, train_data, resume=False):
        if not resume:
            # initialize vocab
            self.vocab = {}
            for sample in train_data:
                sample = sample.lower()
                words = re.findall(self.pat, sample)
                for w in words:
                    # map each utf-8 byte to a unicode char
                    w = ' '.join([self.byte_encoder[b] for b in w.encode('utf-8')])
                    if w in self.vocab:
                        self.vocab[w] += 1
                    else:
                        self.vocab[w] = 1

        # merge
        max_merge_count = 50000
        start = 0
        if resume:
            start = len(self.merge_ops) + 1
        print('starting from merge count: {}, max count: {}'.format(start, max_merge_count))
        for i in range(start, max_merge_count):
            if i % 100 == 0:
                print('training merge count: {}'.format(i))
            # find most frequent pair
            pairs = get_vocab_pairs(self.vocab)
            if len(pairs) == 0:
                break
            best = max(pairs, key=pairs.get)
            # update vocabulary and merge_ops
            self.append_to_vocabulary(''.join(best))
            self.merge_ops.append(best)

            # merge with most frequent pair
            self.vocab = merge_vocab(self.vocab, best)

        print('training final merge count: {}'.format(i))
        print('final vocabulary size: {}'.format(len(self.vocabulary)))

    def token_to_id(self, token):
        if not token in self.vocabulary:
            return None
        return self.vocabulary[token]

    def id_to_token(self, id):
        if id == None:
            return None
        if self.vocabulary_reverse == None or not (len(self.vocabulary_reverse) == len(self.vocabulary)):
            self.vocabulary_reverse = {v: k for k, v in self.vocabulary.items()}
        return self.vocabulary_reverse[id]

    def bpe(self, token):
        # token 'hello'
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
        
        if not pairs:   # single char
            return token
        
        while True:
            bigram = min(pairs, key = lambda pair : self.bpe_ranks(pair))
            if bigram not in self.merge_ops:    # no merge op available
                break
            
            new_word = merge_bigram_in_word(word, bigram)

            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text): # text -> list of token id
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(
                self.byte_encoder[b] for b in token.encode('utf-8') # string -> utf-8 bytes -> unicode string
            )
            bpe_tokens.extend(self.bpe(token).split(" "))
        ids = [self.token_to_id(t) for t in bpe_tokens]
        return bpe_tokens, ids

    def decode(self, ids): # list of token id -> text
        tokens = [self.id_to_token(i) for i in ids]
        text = ''.join(tokens)
        text = bytearray([self.byte_decoder[t] for t in text]).decode('utf-8')  # unicode string -> utf-8 bytes -> string
        return text

if __name__ == '__main__':
    train_data = ['Hello how are you', 'Good afternoon', 'Good morning', 'Good evening', 'state', 'bus', 'business', 'go there']
    test_data = ['How to go to the bus station']

    tokenizer = BPETokenizer()
    tokenizer.train(train_data)
    tokenizer.save('./tmp')
    input = 'hello, i am a robot'
    bpe_tokens, ids = tokenizer.encode(input)
    output = tokenizer.decode(ids)
    print(input)
    print(bpe_tokens)
    print(ids)
    print(output)