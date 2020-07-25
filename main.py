#Text summarization using amazon reviews data. We will be using seq2seq learning with local attention model.
#We will use bidirectional LSTM model for encoding the input sentences and single LSTM layer with an NN to generate the output

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/Processed_Data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


import csv
from nltk import word_tokenize
import string

summaries = []
texts = []

#filtering the input text to remove unwanted characters. Keeping the text_max_len to 500 , min_length to 25 and summary_max_len to 30

def clean(text):
    text = text.lower()
    printable = set(string.printable)
    text = "".join(list(filter(lambda x: x in printable,text))) #filter funny characters, if any.
    return text

text_max_len = 500
text_min_len = 25
summary_max_len = 30
vocab2idx = {}

with open('/kaggle/input/amazon-fine-food-reviews/Reviews.csv') as csvfile:
    Reviews = csv.DictReader(csvfile)
    i=0
    
    for row in Reviews:
        
        text = row['Text']
        summary = row['Summary']
        
        if len(text) <= text_max_len and len(text) >= text_min_len and len(summary) <= summary_max_len:
            #print(i)

            clean_text = clean(text)
            clean_summary = clean(summary)
            
            tokenized_summary = word_tokenize(clean_summary)
            tokenized_text = word_tokenize(clean_text)
            
            # BUILD VOCABULARY
            
            for word in tokenized_text:
                if word not in vocab2idx:
                    vocab2idx[word]=len(vocab2idx)
                    
            for word in tokenized_summary:
                if word not in vocab2idx:
                    vocab2idx[word]=len(vocab2idx)
                    
            ## ________________

            summaries.append(tokenized_summary)
            texts.append(tokenized_text)

            if i%10000==0:
                print("Processing data # {}".format(i))

            i+=1

print("\n# of Data: {}".format(len(texts)))


#Padding the input_text and summary using 'UNK','PAD','EOS'. we will be using glove6b100dtxt word embeddings for our input sentences
vocab = []
embd = []
special_tags = ['<UNK>','<PAD>','<EOS>']


def loadEmbeddings(filename):
    vocab2embd = {}
    
    with open(filename) as infile:     
        for line in infile:
            row = line.strip().split(' ')
            word = row[0].lower()
            if word not in vocab2embd:
                vocab2embd[word]=np.asarray(row[1:],np.float32)

    print('Embedding Loaded.')
    return vocab2embd

vocab2embd = loadEmbeddings('/kaggle/input/glove6b100dtxt/glove.6B.100d.txt')

for word in vocab2idx:
    if word in vocab2embd:
        vocab.append(word)
        embd.append(vocab2embd[word])
        
for special_tag in special_tags:
    vocab.append(special_tag)
    embd.append(np.random.rand(len(embd[0]),))
    
vocab2idx = {word:idx for idx,word in enumerate(vocab)}
embd = np.asarray(embd,np.float32)

print("Vocabulary Size: {}".format(len(vocab2idx)))



vec_texts=[]
vec_summaries=[]

for text,summary in zip(texts,summaries):
    # Replace out of vocab words with index for '<UNK>' tag
    vec_texts.append([vocab2idx.get(word,vocab2idx['<UNK>']) for word in text])
    vec_summaries.append([vocab2idx.get(word,vocab2idx['<UNK>']) for word in summary])
	
	

import random
random.seed(101)

texts_idx = [idx for idx in range(len(vec_texts))]
random.shuffle(texts_idx)

vec_texts = [vec_texts[idx] for idx in texts_idx]
vec_summaries = [vec_summaries[idx] for idx in texts_idx]



#Use first 10000 data for testing, the next 10000 data for validation, and rest for training. Will be feeding the model the data in batches of 32

test_summaries = vec_summaries[0:10000]
test_texts = vec_texts[0:10000]

val_summaries = vec_summaries[10000:20000]
val_texts = vec_texts[10000:20000]

train_summaries = vec_summaries[20000:]
train_texts = vec_texts[20000:]

def bucket_and_batch(texts,summaries,batch_size=32):
    
    # Sort summaries and texts according to the length of text
    # (So that texts with similar lengths tend to remain in the same batch and thus require less padding)
    
    text_lens = [len(text) for text in texts]
    sortedidx = np.flip(np.argsort(text_lens),axis=0)
    texts=[texts[idx] for idx in sortedidx]
    summaries=[summaries[idx] for idx in sortedidx]
    
    batches_text=[]
    batches_summary=[]
    batches_true_text_len = []
    batches_true_summary_len = []
    
    i=0
    while i < (len(texts)-batch_size):
        
        max_len = len(texts[i])
        
        batch_text=[]
        batch_summary=[]
        batch_true_text_len=[]
        batch_true_summary_len=[]
        
        for j in range(batch_size):
            
            padded_text = texts[i+j]
            padded_summary = summaries[i+j]
            
            batch_true_text_len.append(len(texts[i+j]))
            batch_true_summary_len.append(len(summaries[i+j])+1)
     
            while len(padded_text) < max_len:
                padded_text.append(vocab2idx['<PAD>'])

            padded_summary.append(vocab2idx['<EOS>']) #End of Sentence Marker
            while len(padded_summary) < summary_max_len+1:
                padded_summary.append(vocab2idx['<PAD>'])
            
        
            batch_text.append(padded_text)
            batch_summary.append(padded_summary)
        
        batches_text.append(batch_text)
        batches_summary.append(batch_summary)
        batches_true_text_len.append(batch_true_text_len)
        batches_true_summary_len.append(batch_true_summary_len)
        
        i+=batch_size
        
    return batches_text, batches_summary, batches_true_text_len, batches_true_summary_len
	

train_batches_text, train_batches_summary, train_batches_true_text_len, train_batches_true_summary_len \
= bucket_and_batch(train_texts, train_summaries)

val_batches_text, val_batches_summary, val_batches_true_text_len, val_batches_true_summary_len \
= bucket_and_batch(val_texts, val_summaries)

test_batches_text, test_batches_summary, test_batches_true_text_len, test_batches_true_summary_len \
= bucket_and_batch(test_texts, test_summaries)

os.mkdir("/kaggle/Processed_Data")
os.mknod("/kaggle/Processed_Data/Amazon_Reviews_Processed.json")



import json

d = {}

d["vocab"] = vocab2idx
d["embd"] = embd.tolist()
d["train_batches_text"] = train_batches_text
d["test_batches_text"] = test_batches_text
d["val_batches_text"] = val_batches_text
d["train_batches_summary"] = train_batches_summary
d["test_batches_summary"] = test_batches_summary
d["val_batches_summary"] = val_batches_summary
d["train_batches_true_text_len"] = train_batches_true_text_len
d["val_batches_true_text_len"] = val_batches_true_text_len
d["test_batches_true_text_len"] = test_batches_true_text_len
d["train_batches_true_summary_len"] = train_batches_true_summary_len
d["val_batches_true_summary_len"] = val_batches_true_summary_len
d["test_batches_true_summary_len"] = test_batches_true_summary_len

with open('/kaggle/Processed_Data/Amazon_Reviews_Processed.json', 'w') as outfile:
    json.dump(d, outfile)
	
	
import json

with open('/kaggle/Processed_Data/Amazon_Reviews_Processed.json') as file:

    for json_data in file:
        saved_data = json.loads(json_data)

        vocab2idx = saved_data["vocab"]
        embd = saved_data["embd"]
        train_batches_text = saved_data["train_batches_text"]
        test_batches_text = saved_data["test_batches_text"]
        val_batches_text = saved_data["val_batches_text"]
        train_batches_summary = saved_data["train_batches_summary"]
        test_batches_summary = saved_data["test_batches_summary"]
        val_batches_summary = saved_data["val_batches_summary"]
        train_batches_true_text_len = saved_data["train_batches_true_text_len"]
        val_batches_true_text_len = saved_data["val_batches_true_text_len"]
        test_batches_true_text_len = saved_data["test_batches_true_text_len"]
        train_batches_true_summary_len = saved_data["train_batches_true_summary_len"]
        val_batches_true_summary_len = saved_data["val_batches_true_summary_len"]
        test_batches_true_summary_len = saved_data["test_batches_true_summary_len"]

        break
        
idx2vocab = {v:k for k,v in vocab2idx.items()}

#The hidden memory size of LSTM is 300 and window size for our local attention is 5
hidden_size = 300
learning_rate = 0.001
epochs = 5
max_summary_len = 31 # should be summary_max_len as used in data_preprocessing with +1 (+1 for <EOS>) 
D = 5 # D determines local attention window size
window_len = 2*D+1
l2=1e-6

import tensorflow.compat.v1 as tf 
tf.reset_default_graph() 

tf.disable_v2_behavior()
tf.disable_eager_execution()

embd_dim = len(embd[0])

tf_text = tf.placeholder(tf.int32, [None, None])
tf_embd = tf.placeholder(tf.float32, [len(vocab2idx),embd_dim])
tf_true_summary_len = tf.placeholder(tf.int32, [None])
tf_summary = tf.placeholder(tf.int32,[None, None])
tf_train = tf.placeholder(tf.bool)



def dropout(x,rate,training):
    return tf.cond(tf_train,
                    lambda: tf.nn.dropout(x,rate=0.3),
                    lambda: x)
					
					
embd_text = tf.nn.embedding_lookup(tf_embd, tf_text)
embd_summary = tf.nn.embedding_lookup(tf_embd, tf_summary)

embd_text = dropout(embd_text,rate=0.3,training=tf_train)


def LSTM(x,hidden_state,cell,input_dim,hidden_size,scope):
    
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        
        w = tf.get_variable("w", shape=[4,input_dim,hidden_size],
                                    dtype=tf.float32,
                                    trainable=True,
                                    initializer=tf.glorot_uniform_initializer())
        
        u = tf.get_variable("u", shape=[4,hidden_size,hidden_size],
                            dtype=tf.float32,
                            trainable=True,
                            initializer=tf.glorot_uniform_initializer())
        
        b = tf.get_variable("bias", shape=[4,1,hidden_size],
                    dtype=tf.float32,
                    trainable=True,
                    initializer=tf.zeros_initializer())
        
    input_gate = tf.nn.sigmoid( tf.matmul(x,w[0]) + tf.matmul(hidden_state,u[0]) + b[0])
    forget_gate = tf.nn.sigmoid( tf.matmul(x,w[1]) + tf.matmul(hidden_state,u[1]) + b[1])
    output_gate = tf.nn.sigmoid( tf.matmul(x,w[2]) + tf.matmul(hidden_state,u[2]) + b[2])
    cell_ = tf.nn.tanh( tf.matmul(x,w[3]) + tf.matmul(hidden_state,u[3]) + b[3])
    cell = forget_gate*cell + input_gate*cell_
    hidden_state = output_gate*tf.tanh(cell)
    
    return hidden_state, cell
	
S = tf.shape(embd_text)[1] #text sequence length #178
N = tf.shape(embd_text)[0] #batch_size # 32  #embd_text shape is [ 32 178 100]
# embd_summary_shape = tf.shape(embd_summary)



i=0
hidden=tf.zeros([N, hidden_size], dtype=tf.float32)
cell=tf.zeros([N, hidden_size], dtype=tf.float32)
hidden_forward=tf.TensorArray(size=S, dtype=tf.float32)

#shape of embd_text: [N,S,embd_dim]
embd_text_t = tf.transpose(embd_text,[1,0,2]) 
#current shape of embd_text: [S,N,embd_dim]

def cond(i, hidden, cell, hidden_forward):
    return i < S

def body(i, hidden, cell, hidden_forward):
    x = embd_text_t[i]
    
    hidden,cell = LSTM(x,hidden,cell,embd_dim,hidden_size,scope="forward_encoder")
    hidden_forward = hidden_forward.write(i, hidden)

    return i+1, hidden, cell, hidden_forward

_, _, _, hidden_forward = tf.while_loop(cond, body, [i, hidden, cell, hidden_forward])


i=S-1
hidden=tf.zeros([N, hidden_size], dtype=tf.float32)
cell=tf.zeros([N, hidden_size], dtype=tf.float32)
hidden_backward=tf.TensorArray(size=S, dtype=tf.float32)

def cond(i, hidden, cell, hidden_backward):
    return i >= 0

def body(i, hidden, cell, hidden_backward):
    x = embd_text_t[i]
    hidden,cell = LSTM(x,hidden,cell,embd_dim,hidden_size,scope="backward_encoder")
    hidden_backward = hidden_backward.write(i, hidden)

    return i-1, hidden, cell, hidden_backward

_, _, _, hidden_backward = tf.while_loop(cond, body, [i, hidden, cell, hidden_backward])

hidden_forward = hidden_forward.stack()
hidden_backward = hidden_backward.stack()
encoder_states = tf.concat([hidden_forward,hidden_backward],axis=-1)
# encoder_states_shape = tf.shape(encoder_states)  #shape is [178  32 600]
encoder_states = tf.transpose(encoder_states,[1,0,2])
encoder_states_shape = tf.shape(encoder_states) #[ 32 178 600]

encoder_states = dropout(encoder_states,rate=0.3,training=tf_train)

final_encoded_state = dropout(tf.concat([hidden_forward[-1],hidden_backward[-1]],axis=-1),rate=0.3,training=tf_train)
final_encoded_state_shape = tf.shape(final_encoded_state) #shape [ 32 600]


#attention_score score(ht,hs) = htWahs where ht is the current hidden state of output layer and hs is the encoder states
def attention_score(encoder_states,decoder_hidden_state,scope="attention_score"):
    
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        Wa = tf.get_variable("Wa", shape=[2*hidden_size,2*hidden_size], # shape [600, 600]
                                    dtype=tf.float32,
                                    trainable=True,
                                    initializer=tf.glorot_uniform_initializer())
        
    encoder_states = tf.reshape(encoder_states,[N*S,2*hidden_size]) #shape [32*178, 600]
    
    encoder_states = tf.reshape(tf.matmul(encoder_states,Wa),[N,S,2*hidden_size]) #shape [32,178,600]
    decoder_hidden_state = tf.reshape(decoder_hidden_state,[N,2*hidden_size,1]) #shape [32,600,1]
    
    return tf.reshape(tf.matmul(encoder_states,decoder_hidden_state),[N,S]) #shape [32,178]

#In local attention the starting position for attention window is ps = S.sigmoid(v⊤.tanh(Wp.ht)). 
#The center for that window is pt = ps+D. Then a gaussian distribution based position scores are calculated where the position
#close to pt with window size of D will have higher values while calculating the attention scores.
#The gaussian funtion is at(s) = align(ht,h¯s) exp(−(s − pt)2/2σ2). Then the attention score is multiplied with the gaussian_position_scores.

def align(encoder_states, decoder_hidden_state,scope="attention"):
    
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        Wp = tf.get_variable("Wp", shape=[2*hidden_size,128],# pt = S·sigmoid(vp tanh(Wp.ht)) #ht shape [32,600]
                                    dtype=tf.float32,
                                    trainable=True,
                                    initializer=tf.glorot_uniform_initializer())
        
        Vp = tf.get_variable("Vp", shape=[128,1],
                            dtype=tf.float32,
                            trainable=True,
                            initializer=tf.glorot_uniform_initializer())
    
    positions = tf.cast(S-window_len,dtype=tf.float32) # Maximum valid attention window starting position
    
    # Predict attention window starting position 
    sigmoid_values = tf.nn.sigmoid(tf.matmul(tf.tanh(tf.matmul(decoder_hidden_state,Wp)),Vp))
   
    ps = positions*sigmoid_values   #tf.nn.sigmoid(tf.matmul(tf.tanh(tf.matmul(decoder_hidden_state,Wp)),Vp))
    # ps = (soft-)predicted starting position of attention window
    pt = ps+D # pt = center of attention window where the whole window length is 2*D+1
    pt = tf.reshape(pt,[N]) # pt is the point of interest i.e is center next step is to create a gausian dis
    # tribution centered around pt whose size is sentence length so point near to pt will be given more priority
    
    i = 0
    gaussian_position_based_scores = tf.TensorArray(size=S,dtype=tf.float32)
    sigma = tf.constant(D/2,dtype=tf.float32)
    
def cond(i,gaussian_position_based_scores):
        
        return i < S
                      
def body(i,gaussian_position_based_scores):
        
        score = tf.exp(-((tf.square(tf.cast(i,tf.float32)-pt))/(2*tf.square(sigma)))) 
        # (equation (10) in https://nlp.stanford.edu/pubs/emnlp15_attn.pdf)
        gaussian_position_based_scores = gaussian_position_based_scores.write(i,score)
            
        return i+1,gaussian_position_based_scores
    
    # Looping to calculate individual value of the gaussian distribution where it takes current position and pt 
                      
    i,gaussian_position_based_scores = tf.while_loop(cond,body,[i,gaussian_position_based_scores])
    
    gaussian_position_based_scores = gaussian_position_based_scores.stack()
    gaussian_position_based_scores = tf.transpose(gaussian_position_based_scores,[1,0])
    gaussian_position_based_scores = tf.reshape(gaussian_position_based_scores,[N,S])
    
    #attention score shape [32,178] and gaussian position score is [32,178] where position score around D distance from pt
    # have high values
    scores = attention_score(encoder_states,decoder_hidden_state)*gaussian_position_based_scores
    scores = tf.nn.softmax(scores,axis=-1)
    
    return tf.reshape(scores,[N,S,1])

#Based on the tf_train which is True if we are training the model and False is we are validating it,
#we will feed the input to the decoder. During training we will be using teacher forcing method where
#we feed the correct next word as input instead of the previous predicted word, so that the network will learn. 
#While validating we will feed the previously predicted word as the input for the next time step.

def nextInput(embd_summary_t, i,tf_embd,next_word_vec,N,embd_dim):   

	    return tf.cond(tf_train,
                    lambda: embd_summary_t[i],
                    lambda: tf.reshape(tf.nn.embedding_lookup(tf_embd, next_word_vec),[N, embd_dim]))
	
	
#SOS represents starting marker. It tells the decoder that it is about to decode the first word of the output. SOS is a trainable parameter
with tf.variable_scope("decoder",reuse=tf.AUTO_REUSE):
    SOS = tf.get_variable("sos", shape=[1,embd_dim],
                                dtype=tf.float32,
                                trainable=True,
                                initializer=tf.glorot_uniform_initializer())
    
    # SOS represents starting marker 
    # It tells the decoder that it is about to decode the first word of the output
    # I have set SOS as a trainable parameter
    
    Wc = tf.get_variable("Wc", shape=[2*hidden_size,embd_dim],
                            dtype=tf.float32,
                            trainable=True,
                            initializer=tf.glorot_uniform_initializer())
    


SOS = tf.tile(SOS,[N,1]) #now SOS shape: [N,embd_dim]
inp = SOS
hidden=final_encoded_state
cell=tf.zeros([N, 2*hidden_size], dtype=tf.float32)
decoder_outputs=tf.TensorArray(size=max_summary_len, dtype=tf.float32)
outputs=tf.TensorArray(size=max_summary_len, dtype=tf.int32)

embd_summary_t = tf.transpose(embd_summary,[1,0,2])

#encoder_context_vector shape is [32,600]
#The above encoder_context_vector is simple multiplication not matmul where the scores like 0.3 is multiplied for one word hidden state and so on
for i in range(max_summary_len):
    
    attention_scores = align(encoder_states,hidden)
    encoder_context_vector = tf.reduce_sum(encoder_states*attention_scores,axis=1) #[32,178,600]* #[32 178 1] #shape [ 32 600]
    inp = dropout(inp,rate=0.3,training=tf_train)
    
    inp = tf.concat([inp,encoder_context_vector],axis=-1)
    inp_shape = tf.shape(inp)
    
    #LSTM(x,hidden_state,cell,input_dim,hidden_size,scope)
    hidden,cell = LSTM(inp,hidden,cell,embd_dim+2*hidden_size,2*hidden_size,scope="decoder")
    
    hidden = dropout(hidden,rate=0.3,training=tf_train)  
    
    
    linear_out = tf.nn.tanh(tf.matmul(hidden,Wc)) #Wc shape [1200,100]
    
    linear_out_shape = tf.shape(linear_out)     # [ 32 100]
    decoder_output = tf.matmul(linear_out,tf.transpose(tf_embd,[1,0])) #[emd_dim,len(vocab2idx)]
    # produce unnormalized probability distribution over vocabulary
    
    decoder_output_shape = tf.shape(decoder_output) #[32 43546]
    
    
    decoder_outputs = decoder_outputs.write(i,decoder_output)
    
    # Pick out most probable vocab indices based on the unnormalized probability distribution
    
    next_word_vec = tf.cast(tf.argmax(decoder_output,1),tf.int32)

    next_word_vec = tf.reshape(next_word_vec, [N])

    outputs = outputs.write(i,next_word_vec)
    inp = nextInput(embd_summary_t, i,tf_embd,next_word_vec,N,embd_dim)
    
    

         
    
    
decoder_outputs = decoder_outputs.stack()
outputs = outputs.stack()

decoder_outputs = tf.transpose(decoder_outputs,[1,0,2])
outputs = tf.transpose(outputs,[1,0])

filtered_trainables = [var for var in tf.trainable_variables() if
                       not("Bias" in var.name or "bias" in var.name
                           or "noreg" in var.name)]

regularization = tf.reduce_sum([tf.nn.l2_loss(var) for var
                                in filtered_trainables])

#we will be using sparse_softmax_cross_entropy as the cost function and for accuracy
#we will be comparing the predicted sentence with the actual output.

with tf.variable_scope("loss"):

    epsilon = tf.constant(1e-9, tf.float32)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf_summary, logits=decoder_outputs)

    pad_mask = tf.sequence_mask(tf_true_summary_len,
                                maxlen=max_summary_len,
                                dtype=tf.float32)

    masked_cross_entropy = cross_entropy*pad_mask

    cost = tf.reduce_mean(masked_cross_entropy) + \
        l2*regularization

    cross_entropy = tf.reduce_mean(masked_cross_entropy)
	
# Comparing predicted sequence with labels
comparison = tf.cast(tf.equal(outputs, tf_summary),
                     tf.float32)

# Masking to ignore the effect of pads while calculating accuracy
pad_mask = tf.sequence_mask(tf_true_summary_len,
                            maxlen=max_summary_len,
                            dtype=tf.bool)

masked_comparison = tf.boolean_mask(comparison, pad_mask)

# Accuracy
accuracy = tf.reduce_mean(masked_comparison)



all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

gvs = optimizer.compute_gradients(cost, all_vars)

capped_gvs = [(tf.clip_by_norm(grad, 5), var) for grad, var in gvs] # Gradient Clipping

train_op = optimizer.apply_gradients(capped_gvs)


import pickle
import random

with tf.Session() as sess:  # Start Tensorflow Session
    display_step = 25
    
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(tf.tables_initializer())

    epoch=0
    while epoch<epochs:
        
        print("\n\nSTARTING TRAINING\n\n")
        
        batches_indices = [i for i in range(0, len(train_batches_text))]
        random.shuffle(batches_indices)

        total_train_acc = 0
        total_train_loss = 0

        for i in range(0, len(train_batches_text)):
            
            j = int(batches_indices[i])

            cost,prediction,\
                acc, _ = sess.run([cross_entropy,
                                   outputs,
                                   accuracy,
                                   train_op],
                                  feed_dict={tf_text: train_batches_text[j],
                                             tf_embd: embd,
                                             tf_summary: train_batches_summary[j],
                                             tf_true_summary_len: train_batches_true_summary_len[j],
                                             tf_train: True})
            
            total_train_acc += acc
            total_train_loss += cost

            if i % display_step == 0:
                print("Iter "+str(i)+", Cost= " +
                      "{:.3f}".format(cost)+", Acc = " +
                      "{:.2f}%".format(acc*100))
            
            if i % 500 == 0:
                
                idx = random.randint(0,len(train_batches_text[j])-1)
                
                
                
                text = " ".join([idx2vocab.get(vec,"<UNK>") for vec in train_batches_text[j][idx]])
                predicted_summary = [idx2vocab.get(vec,"<UNK>") for vec in prediction[idx]]
                actual_summary = [idx2vocab.get(vec,"<UNK>") for vec in train_batches_summary[j][idx]]
                
                print("\nSample Text\n")
                print(text)
                print("\nSample Predicted Summary\n")
                for word in predicted_summary:
                    if word == '<EOS>':
                        break
                    else:
                        print(word,end=" ")
                print("\n\nSample Actual Summary\n")
                for word in actual_summary:
                    if word == '<EOS>':
                        break
                    else:
                        print(word,end=" ")
                print("\n\n")
                
        print("\n\nSTARTING VALIDATION\n\n")
                
        total_val_loss=0
        total_val_acc=0
                
        for i in range(0, len(val_batches_text)):
            
            if i%100==0:
                print("Validating data # {}".format(i))

            cost, prediction,\
                acc = sess.run([cross_entropy,
                                outputs,
                                accuracy],
                                  feed_dict={tf_text: val_batches_text[i],
                                             tf_embd: embd,
                                             tf_summary: val_batches_summary[i],
                                             tf_true_summary_len: val_batches_true_summary_len[i],
                                             tf_train: False})
            
            total_val_loss += cost
            total_val_acc += acc
            
        avg_val_loss = total_val_loss/len(val_batches_text)
        
        print("\n\nEpoch: {}\n\n".format(epoch))
        print("Average Training Loss: {:.3f}".format(total_train_loss/len(train_batches_text)))
        print("Average Training Accuracy: {:.2f}".format(100*total_train_acc/len(train_batches_text)))
        print("Average Validation Loss: {:.3f}".format(avg_val_loss))
        print("Average Validation Accuracy: {:.2f}".format(100*total_val_acc/len(val_batches_text)))
                    
              
        epoch+=1
