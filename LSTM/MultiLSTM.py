# -*- coding: utf-8 -*-
'''
Created on 2021年2月22日

@author: xiaoyw
'''
# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
Run this script on tensorflow r0.10. Errors appear when using lower versions.
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


BATCH_START = 0
TIME_STEPS = 15
BATCH_SIZE = 30
INPUT_SIZE = 25
OUTPUT_SIZE = 4
PRED_SIZE = 15 #预测输出1天序列数据
CELL_SIZE = 256
NUM_LAYERS = 3
LR = 0.0001
EPOSE = 30000
dropout = 0.2 #?

def get_train_data():
    df = pd.read_csv('share20210306.csv')
    return df

def get_test_data():
    df = pd.read_csv('share20210306.csv')
    #df = df.iloc[-(TIME_STEPS+1):] #由于要删除一行，需要多取一行（美国股市与中国差一天）
    df = df.iloc[-TIME_STEPS:]
    return df

def get_pred_data(y,z,sc):
    yy = np.concatenate((y, z),axis=1)
    y=sc.inverse_transform(yy)
    return y

# 设置数据集    
def set_datas(df,train=True,sc=None):        
    df['Year'] = df['trade_date'].apply(lambda x:int(str(x)[0:4]))
    df['Month'] = df['trade_date'].apply(lambda x:int(str(x)[4:6]))
    df['Day'] = df['trade_date'].apply(lambda x:int(str(x)[6:8]))

    df['Week'] = df['trade_date'].apply(lambda x:datetime.datetime.strptime(str(x),'%Y%m%d').weekday())
    #纳仕达克、道琼斯指数，需要下移一条记录
    #shift_columns = ['open3','high3','close3','low3','change3','pct_chg3','open4','high4','close4','low4','change4','pct_chg4']
    #df[shift_columns] = df[shift_columns].shift(1)
    # 重排表的列，便于数据提取
    ##df = df.reindex(columns=col_name)
    df = df.drop('trade_date',axis=1)
    #df = df[1:].reset_index(drop=True)  #删除第一行，从0开始
    col_name = df.columns.tolist()
    #列移动归集位置
    col_name.remove('close0')
    col_name.remove('close1')
    col_name.remove('close2')
    col_name.remove('vol0')
    #删除不重要的列
    #del_list = ['high3','low3','change3','pct_chg3','high4','low4','change4','pct_chg4','high5','low5','change5','pct_chg5']
    #for name in del_list:
    #    col_name.remove(name)    
    col_name.insert(0,'close1')
    col_name.insert(1,'close2')
    col_name.insert(2,'close0')
    col_name.insert(3,'vol0')
    df = df[col_name]
    #sc = MinMaxScaler(feature_range= (0,1)) 预测值超过最大值？
    if train:
        sc = MinMaxScaler(feature_range= (0,1))
        training_set = sc.fit_transform(df)
    else:
        # 测试集，也需要使用原Scaler归一化
        training_set = sc.transform(df)
    # 按时序长度构造数据集
    def get_batch(train_x,train_y):
        data_len = len(train_x) - TIME_STEPS
        seq = []
        res = []
        for i in range(data_len):
            seq.append(train_x[i:i + TIME_STEPS])
            res.append(train_y[i:i + TIME_STEPS]) #取后5组数据
            #res.append(train_y[i:i + TIME_STEPS]) 
         
        seq ,res = np.array(seq),np.array(res)
    
        return  seq, res
    
    if train:
        seq, res = get_batch(training_set[:-PRED_SIZE], training_set[PRED_SIZE:][:,0:OUTPUT_SIZE]) #0:9
    else:
        seq, res = training_set, training_set[:,0:OUTPUT_SIZE]
        seq, res = seq[np.newaxis,:,:], res[np.newaxis,:,:]
  
    return seq, res, training_set[:,OUTPUT_SIZE:],sc,col_name,df

class MultiLSTM(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size,num_layers,is_training):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size # LSTM神经单元数
        
        self.batch_size_ = batch_size # 输入batch_size大小
        self.num_layers = num_layers # LSTM层数
        #是否是训练状态
        self.is_training = is_training
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
            self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
            #节点不被dropout的概率
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('Multi_LSTM'):
            self.add_multi_cell()            
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    def add_input_layer(self,):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size,])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')
        
    # 定义多层LSTM    
    def add_multi_cell(self):
        cell_list = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True) 

        with tf.name_scope('dropout'):
            if self.is_training:
                # 添加dropout.为了防止过拟合，在它的隐层添加了 dropout 正则
                cell_list = tf.contrib.rnn.DropoutWrapper(cell_list, output_keep_prob=self.keep_prob)
                tf.summary.scalar('dropout_keep_probability', self.keep_prob)
        
        lstm_cell = [cell_list for _ in range(self.num_layers)]
        lstm_cell = tf.contrib.rnn.MultiRNNCell(lstm_cell, state_is_tuple=True) #遗漏过？, state_is_tuple=True

        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        # 初始state,全部为0，慢慢的累加记忆
        # time_major -- 默认false,inputs 和outputs 张量的形状格式。如果为True，则这些张量都应该是（都会是）[max_time, batch_size, hidden_size]。如果为false，则这些张量都应该是（都会是）[batch_size，max_time, hidden_size]。time_major=true说明输入和输出tensor的第一维是max_time。否则为batch_size。                     
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)
     
    # 定义输出全连接层
    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        #outputs = tf.unstack(tf.transpose(self.cell_outputs, [1,0,2])) # 
        #self.cell_outputs = tf.contrib.layers.fully_connected(self.cell_outputs, [-1,5,9], None)
        #print('self.cell_outputs shape is {}'.format(self.cell_outputs.shape))        
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        #l_out_x = tf.reshape(self.cell_outputs[:,-1,:], [-1, self.cell_size], name='2_2D')
        
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

    def compute_cost(self):
        #self.cost = tf.losses.mean_squared_error(labels = self.ys,predictions = self.pred)
        # 均方误差
        '''
        #losses= tf.reduce_mean(tf.square(tf.reshape(self.pred, [-1]) - tf.reshape(self.ys, [-1])))

        #losses = tf.losses.mean_squared_error(labels = tf.reshape(self.ys, [-1], name='reshape_target'),predictions = tf.reshape(self.pred, [-1], name='reshape_pred'))        
        losses= tf.reduce_mean(tf.abs(tf.reshape(self.pred, [-1]) - tf.reshape(self.ys, [-1])),name='losses')
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                losses,
                self.batch_size_,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)        
        '''
        #logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
        #targets: List of 1D batch-sized int32 Tensors of the same length as logits.
        #weights: List of 1D batch-sized float-Tensors of the same length as logits.
        #return:log_pers 形状是 [batch_size].
        
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')], 
            [tf.reshape(self.ys, [-1], name='reshape_target')],       
            [tf.ones([self.batch_size * self.n_steps*self.output_size], dtype=tf.float32)], 
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        
        with tf.name_scope('average_cost'):
            self.cost = tf.reduce_max(losses, name='average_cost')
            '''
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size_,
                name='average_cost')
            '''
            tf.summary.scalar('cost', self.cost)
        print('self.cost shape is {}'.format(self.cost.shape))
        

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)


if __name__ == '__main__':

    model = MultiLSTM(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, NUM_LAYERS,True)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs", sess.graph)
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir='logs'
    state = 0
    xs = 0
    df = get_train_data()
    train_x,train_y,z,sc,col_name,df = set_datas(df,True)
    
    # 使用from_tensor_slices将数据放入队列，使用batch和repeat划分数据批次，且让数据序列无限延续
    dataset = tf.data.Dataset.from_tensor_slices((train_x,train_y))
    dataset = dataset.batch(BATCH_SIZE).repeat()
    
    # 使用生成器make_one_shot_iterator和get_next取数据
    # 单次迭代器只能循环使用一次数据，而且单次迭代器不需要手动显示调用sess.run()进行初始化即可使用
    #iterator = dataset.make_one_shot_iterator()
    # 可初始化的迭代器可以重新初始化进行循环，但是需要手动显示调用sess.run()才能循环
    iterator = dataset.make_initializable_iterator()
    next_iterator = iterator.get_next()    
    losse = []
    for i in range(EPOSE):
        # 这是显示初始化，当我们的迭代器是dataset.make_initializable_iterator()的时候，才需要调用这个方法，否则不需要
        sess.run(iterator.initializer)
        seq, res = sess.run(next_iterator)
        #print('train seq.shape is : ',seq.shape)

        if i == 0:
            feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    model.batch_size:BATCH_SIZE,
                    model.keep_prob:0.75,
                    # create initial state
            }
        else:
            feed_dict = {
                model.xs: seq,
                model.ys: res,
                model.batch_size:BATCH_SIZE,
                model.keep_prob:0.75,
                model.cell_init_state: state    # use last state as the initial state for this run
            }

        _, cost, state, pred = sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred],
            feed_dict=feed_dict)
        #if i>40:
        losse.append(cost)

        if i % 20 == 0:
            #print(state)
            print('cost: ', round(cost, 5))
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)
    plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
    plt.rcParams['axes.unicode_minus']=False
    losse = np.array(losse)/max(losse)
    plt.plot(losse, label='Training Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()
    #训练结束
    #--------------
    df = get_test_data()
    seq,res,z,sc,col_name,df = set_datas(df,False,sc) 
    seq = seq.reshape(-1,TIME_STEPS,INPUT_SIZE)
    share_close = df['close0'].values
    share_vol = df['vol0'].values/10000
    share_sh = df['close1'].values
    share_sz = df['close2'].values
    model.is_training = False

    feed_dict = {
        model.xs: seq,
        model.batch_size:1,
        model.keep_prob:1.0,
        #model.cell_init_state: state    # use last state as the initial state for this run
    } 
    #pred,state = sess.run([model.pred,model.cell_init_state], feed_dict=feed_dict)
    pred = sess.run([model.pred], feed_dict=feed_dict)  
    #print(pred[0])
    y=get_pred_data(pred[0].reshape(TIME_STEPS,OUTPUT_SIZE),z,sc) 
    df= pd.DataFrame(y,columns=col_name)   
    df.to_csv('y.csv') 
    share_close1 = df['close0'].values
    share_vol1 = df['vol0'].values/10000
    share_sh1 = df['close1'].values
    share_sz1 = df['close2'].values
    #合并预测移动移位PRED_SIZE
    share_close1 = np.concatenate((share_close[:PRED_SIZE],share_close1),axis=0)
    share_vol1 = np.concatenate((share_vol[:PRED_SIZE],share_vol1),axis=0)
    share_sh1 = np.concatenate((share_sh[:PRED_SIZE],share_sh1),axis=0)
    share_sz1 = np.concatenate((share_sz[:PRED_SIZE],share_sz1),axis=0)
    
    plt.plot(share_sh, label='收盘沪指指数')
    plt.plot(share_sh1, label='预测收盘沪指指数')
    plt.plot(share_sz, label='收盘深证指数')
    plt.plot(share_sz1, label='预测收盘深证指数')
        
    plt.plot(share_close, label='收盘实际值')
    plt.plot(share_vol, label='成交量实际值')
    plt.plot(share_vol1, label='成交量预测值')
    plt.plot(share_close1, label='收盘预测值')
        
    plt.title('Test Loss')
    plt.legend()
    plt.show()        
