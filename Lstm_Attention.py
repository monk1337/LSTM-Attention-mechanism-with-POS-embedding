
# coding: utf-8



import tensorflow as tf
import numpy as np




tf.reset_default_graph()

class lstm_Attention_with_Pos(object):
    
    
    def __init__(self,vocab_size,embedding_dim,num_cell,dropout_value,pos_vocab_size,pdim,category_size):
        
        
        
        
        sentence = tf.placeholder(name='input_data',shape=[None,None],dtype=tf.int32)
        pos_     = tf.placeholder(name='pos',shape=[None,None],dtype=tf.int32)
        intents  = tf.placeholder(name='labels',shape=[None],dtype=tf.int32)
        mode =  tf.placeholder(name='dropout',shape=(),dtype=tf.int32)
        
        
        
        
        
        self.placeholders = {
            
                            'input_d':sentence,
                            'labels':intents,
                             'pos':pos_,
                            'dropout':mode
                            }
        

        
        
        embedding_matrix = tf.get_variable(name='word_embedding',
                                           shape=[vocab_size,embedding_dim],
                                           dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(-0.01,0.01))
        
        embedd_word2vec = tf.nn.embedding_lookup(embedding_matrix,sentence)
        
        dropout = tf.cond(
                          tf.equal(mode,0),
                          lambda : dropout_value, 
                          lambda : 0.
                         )
        
        pos_embedding = tf.get_variable(name='pos_embedding',
                                           shape=[pos_vocab_size,pdim],
                                           dtype=tf.float32,
                                           initializer=tf.random_uniform_initializer(-0.01,0.01))
        
        
        pos_lookup = tf.nn.embedding_lookup(pos_embedding, pos_ )
        
        final_input = tf.concat([embedd_word2vec,pos_lookup],axis=-1)
        
        
                
        
        
        
        
        sequence_length = tf.count_nonzero(sentence,axis=-1)
        
        
         
        lstm_cell_fw = tf.contrib.rnn.LSTMCell(num_cell,forget_bias=1.0)
        dropout_wrapper_fw = tf.contrib.rnn.DropoutWrapper(lstm_cell_fw,output_keep_prob=1. - dropout_value)
        
        
        lstm_cell_bw = tf.contrib.rnn.LSTMCell(num_cell,forget_bias=1.0)
        dropout_wrapper_bw = tf.contrib.rnn.DropoutWrapper(lstm_cell_bw,output_keep_prob=1.- dropout_value)
        
        
        output , last_state     = tf.nn.bidirectional_dynamic_rnn(dropout_wrapper_fw,
                                                      dropout_wrapper_bw,
                                                      final_input,
                                                      sequence_length=sequence_length,
                                                      dtype=tf.float32)
        
        logits = tf.concat(output,2)
        #ex . 12 x 10 x 24
        
        
        
        
        #Attention_layer
        
        
        #ex : 12x10x24 . ===> 120 x 24
        input_reshape = tf.reshape(logits,[-1,num_cell*2])
        
        
        #num_cell = 12
        #ex : # 24 x 1
        attention_size = tf.get_variable(name='attention_size',
                                         shape=[2*num_cell,1],
                                         dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-0.01,0.01))
        # bias 1
        bias          = tf.get_variable(name='bias',shape=[1],
                                        dtype=tf.float32,
                                        initializer=tf.random_uniform_initializer(-0.01,0.01))
        
        
        #projection without activation 
        #ex : 120x24 matmul 24x 1 ==> 120x1
        attention_projection = tf.add(tf.matmul(input_reshape,attention_size),bias)
        
        
        #reshape . 120x1 ==> 12x10x1 (shape of input )
        output_reshape = tf.reshape(attention_projection,[tf.shape(sentence)[0],tf.shape(sentence)[1],-1])
        
        #softmax over logits 12x10x1
        attention_output = tf.nn.softmax(output_reshape,dim=1)
        
        
        #reshape as input 12x10
        attention_visualize = tf.reshape(attention_output,
                                         [tf.shape(sentence)[0],
                                          tf.shape(sentence)[1]],
                                         name='Plot')
        
        
        # 12x10x1 multiply 12x10x24  == > 12x10x24
        attention_projection_output = tf.multiply(attention_output,logits)
        
        #reduce across time 120x10x24 ==> 12x24
        Final_output = tf.reduce_sum(attention_projection_output,1)
        
        
        
        #fully_connected_layer
        
        f_c = tf.get_variable(name='f_c_layer',
                              shape=[2*num_cell,category_size],
                              dtype=tf.float32,
                              initializer=tf.random_uniform_initializer(-0.01,0.01))
        
        
        f_bias = tf.get_variable(name='fc_bias',
                                 shape=[category_size],
                                dtype=tf.float32,
                                 initializer=tf.random_uniform_initializer(-0.01,0.01))
        
        output_projection = tf.add(tf.matmul(Final_output,f_c),f_bias)
        
        
        
        probability_dist  = tf.nn.softmax(output_projection,name='pred')
        
        prediction  = tf.argmax(probability_dist,axis=-1)
        
        
        #as_usual_stuff_
        cross_entropy  = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=intents,logits=output_projection)
        loss = tf.reduce_mean(cross_entropy)
        accuracy_calculation = tf.reduce_mean(tf.cast(tf.equal(tf.cast(prediction,tf.int32),intents),tf.float32))
        
        
        
        self.output = {  'loss':loss,
                         'accuracy':accuracy_calculation,
                         'prediction':prediction,
                         'probai':probability_dist,
                         'logits':output_projection,
                         'attention_visualize':attention_visualize
            
                       }
        
        
        # this guy is hero of this movie
        self.train = tf.train.AdamOptimizer().minimize(loss)
        

def execute_model(model):
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        
        for i in range(1000):
            outp,train = sess.run([model.output,model.train],feed_dict={model.placeholders['input_d']:np.random.randint(0,3,[12,10]),
                                                model.placeholders['labels']:np.random.randint(0,2,[12,]),
                                                model.placeholders['dropout']:0,
                                                model.placeholders['pos']: np.random.randint(0,3,[12,10]) })
            print(outp['loss'],outp['accuracy'],outp['prediction'].shape,outp['attention_visualize'].shape)
        
        
if __name__ == '__main__':
    
    model = lstm_Attention_with_Pos(130,100,12,0.5,120,22,12)
    execute_model(model)
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        

