# -*- coding: utf-8 -*-
'''
Created on 2021年2月22日

@author: xiaoyw
'''

import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 模型参数定义
BATCH_START = 0
TIME_STEPS = 35 #时序长度
BATCH_SIZE = 20 #batch尺寸
INPUT_SIZE = 22 #每个时序点输入数据数量
OUTPUT_SIZE = 4 #每个时序点输出数据数量
PRED_SIZE = 15 #预测输出15天序列数据
CELL_SIZE = 200 #神经元数量
NUM_LAYERS = 3  #LSTM层数
LR = 0.001     #学习率
EPOSE = 4000   #训练次数
keep_prob=0.75
dropout = 0.2 #?

# 输入输出文件名称
Input_file = 'share002230_0328.csv'
Out_flie = 'y002230_0328a.csv'

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
            self.train_optimizer()
            #self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    def add_input_layer(self,):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size,])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_in'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
            tf.summary.histogram('Ws_in',Ws_in)
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')
        
    # 定义多层LSTM    
    def add_multi_cell(self):
        #cell_list = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True) 
        #更换默认tanh激活函数
        cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True, activation=tf.nn.relu) 

        with tf.name_scope('dropout'):
            if self.is_training:
                # 添加dropout.为了防止过拟合，在它的隐层添加了 dropout 正则
                keep_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
                tf.summary.scalar('dropout_keep_probability', self.keep_prob)
            else:
                keep_cell = cell
        
        cell_list = [keep_cell for _ in range(self.num_layers)]
        lstm_cell = tf.contrib.rnn.MultiRNNCell(cell_list, state_is_tuple=True) #遗漏过？, state_is_tuple=True

        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            tf.summary.histogram('initial_state',self.cell_init_state)
        # 初始state,全部为0，慢慢的累加记忆
        # time_major -- 默认false,inputs 和outputs 张量的形状格式。如果为True，则这些张量都应该是（都会是）[max_time, batch_size, hidden_size]。如果为false，则这些张量都应该是（都会是）[batch_size，max_time, hidden_size]。time_major=true说明输入和输出tensor的第一维是max_time。否则为batch_size。                     
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)
     
        #with tf.name_scope('batch_normal'):
        #    self.cell_outputs = tf.layers.batch_normalization(self.cell_outputs, training=True) # momentum=0.4,
     
    # 定义输出全连接层
    def add_output_layer(self):
        # shape = (batch * steps, cell_size)  
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        #l_out_x = tf.reshape(self.cell_outputs[:,-1,:], [-1, self.cell_size], name='2_2D')
        
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_out'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out
            tf.summary.histogram('Wx_plus_out',self.pred)
            tf.summary.histogram('Ws_out',Ws_out)

    def compute_cost(self):
        # 均方误差
        #tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值
        #losses= tf.reduce_mean(tf.abs(tf.reshape(self.pred, [-1]) - tf.reshape(self.ys, [-1])),name='losses')
        losses = tf.losses.mean_squared_error(tf.reshape(self.pred, [-1]) , tf.reshape(self.ys, [-1]))
        with tf.name_scope('average_cost'):
            self.cost = losses
            tf.summary.scalar('cost', self.cost)        

        '''
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')], 
            [tf.reshape(self.ys, [-1], name='reshape_target')],       
            [tf.ones([self.batch_size * self.n_steps*self.output_size], dtype=tf.float32)], 
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        
        with tf.name_scope('average_cost'):
            #self.cost = tf.reduce_max(losses, name='average_cost')           
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size_,
                name='average_cost')            
            tf.summary.scalar('cost', self.cost)
        print('self.cost shape is {}'.format(self.cost.shape))
        '''
    
    def train_optimizer(self):   
        # 使用Adam梯度下降
        optimizer = tf.train.AdamOptimizer(LR) #,beta1=0.9)
        # 裁剪一下Gradient输出，最后的gradient都在[-1, 1]的范围内
        # 计算导数   cost为损失函数
        gradients = optimizer.compute_gradients(self.cost)
        # 限定导数值域-1到1
        #capped_gradients = [(tf.clip_by_value(grad, -1., 1.0), var) for grad, var in gradients if grad is not None]
        capped_gradients = [(tf.clip_by_value(grad, -0.5, 0.5), var) for grad, var in gradients if grad is not None]
        #self.p =tf.print(capped_gradients,[capped_gradients],'Adam_clip')        
        # 将处理后的导数继续应用到LSTM中
        self.train_op = optimizer.apply_gradients(capped_gradients) 
        # 基于损失函数w.r.t的梯度值实现停止条件
        grad_norms = [tf.nn.l2_loss(grad) for grad, var in gradients]
        self.grad_norm = tf.add_n(grad_norms) 
        #grad_norm = [tf.reduce_sum(grad) for grad, var in gradients] 
        #self.grad_norm = tf.reduce_sum(grad_norm)     

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=0.1,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.01)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)
#建立LSTM模型代码结束


#以下是训练模型部分代码
def get_train_data():
    df = pd.read_csv(Input_file)
    return df

def get_test_data():
    df = pd.read_csv(Input_file)
    df = df.iloc[-TIME_STEPS:]
    return df

def get_pred_data(y,z,sc):
    yy = np.concatenate((y, z),axis=1)
    y=sc.inverse_transform(yy)
    return y

# 设置数据集    
def set_datas(df,train=True,sc=None):        

    col_name = df.columns.tolist()
    df['Year'] = df['trade_date'].apply(lambda x:int(str(x)[0:4]))
    df['Month'] = df['trade_date'].apply(lambda x:int(str(x)[4:6]))
    df['Day'] = df['trade_date'].apply(lambda x:int(str(x)[6:8]))

    df['Week'] = df['trade_date'].apply(lambda x:datetime.datetime.strptime(str(x),'%Y%m%d').weekday())   
    #列移动归集位置
    trade_train = ['close0','close1','close2','vol0']
    
    #列移动归集位置，方便获取y值，从头连续
    for col in trade_train:
        col_name.remove(col)
        col_name.insert(0,col)

    df = df[col_name]
    #sc = MinMaxScaler(feature_range= (0,1)) 预测值超过最大值？
    if train:
        sc = MinMaxScaler(feature_range= (0,1))
        training_set = sc.fit_transform(df)
    else:
        # 测试集，也需要使用原Scaler归一化
        if sc==None:
            sc = MinMaxScaler(feature_range= (0,1))
            training_set = sc.fit_transform(df)            
        else:
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
    #返回训练集
    if train:
        seq, res = get_batch(training_set[:-PRED_SIZE], training_set[PRED_SIZE:][:,0:OUTPUT_SIZE]) #0:9
    #返回测试集，一条记录
    else:
        seq, res = training_set, training_set[:,0:OUTPUT_SIZE]
        seq, res = seq[np.newaxis,:,:], res[np.newaxis,:,:]
  
    return seq, res, training_set[:,OUTPUT_SIZE:],sc,col_name,df



# 训练部分代码
# 输入model，建模过程放在主程序中，与测试部分共享模型代码
# 输入sess和saver原理同上
def train(model,sess,saver,merged,init):
    writer = tf.summary.FileWriter("logs", sess.graph)
    sess.run(init)
    # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
    # $ tensorboard --logdir logs
    state = 0
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

        if i == 0:
            feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    model.batch_size:BATCH_SIZE,
                    model.keep_prob:keep_prob,
                    # create initial state
            }
        else:
            feed_dict = {
                model.xs: seq,
                model.ys: res,
                model.batch_size:BATCH_SIZE,
                model.keep_prob:keep_prob,
                model.cell_init_state: state    # use last state as the initial state for this run
            }

        _, cost, state,gn = sess.run(
            [model.train_op, model.cost, model.cell_final_state,model.grad_norm ],
            feed_dict=feed_dict)

        losse.append(cost)        

        if i % 20 == 0:                     
            print('i: ', i,' , cost: ', round(cost, 5))
            #print('grad:' ,round(gn,5))
            result = sess.run(merged,feed_dict)
            writer.add_summary(result, i)
            
        # 基于损失函数w.r.t的梯度值实现停止条件
        some_treshold = 0.00001
        if(gn < some_treshold): 
            print("Training finished.") 
            break 
        
    saver.save(sess, 'model_save1\modle.ckpt')

    plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
    plt.rcParams['axes.unicode_minus']=False
    #losse = np.array(losse)/max(losse)
    plt.plot(losse, label='Training Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()
# 训练结束
# 默认TEST是同训练同步执行
def test(model,sess,saver,TEST = False):
    df = get_test_data()
    seq,res,z,sc,col_name,df = set_datas(df,False) 
    seq = seq.reshape(-1,TIME_STEPS,INPUT_SIZE)
    trade_train = ['close0','close1','close2','vol0']
    trade_init = []
    for col in trade_train:
        trade_init.append(df[col].values)

    if TEST:
        saver.restore(sess,tf.train.latest_checkpoint('model_save1'))

    model.is_training = False

    feed_dict = {
        model.xs: seq,
        model.batch_size:1,
        model.keep_prob:1.0,
        #model.cell_init_state: state    # use last state as the initial state for this run
    } 
    #pred,state = sess.run([model.pred,model.cell_init_state], feed_dict=feed_dict)
    pred,_ = sess.run([model.pred, model.cell_final_state], feed_dict=feed_dict)  
    #print(pred[0])
    y=get_pred_data(pred,z,sc) 
    #y=get_pred_data(pred[0].reshape(TIME_STEPS,OUTPUT_SIZE),z,sc) 
    df= pd.DataFrame(y,columns=col_name)   
    df.to_csv(Out_flie) 
    trade_pred = []
    i = 0
    for col in trade_train:
        trade_pred.append(np.concatenate((trade_init[i][:PRED_SIZE],df[col].values),axis=0))
        i =i +1
    #合并预测移动移位PRED_SIZE
    
    plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
    plt.rcParams['axes.unicode_minus']=False    
    lables = ['收盘预测值','收盘沪指指数','收盘深证指数','成交量预测值']
    for i in range(4):
        plt.subplot(2,2,(i+1))
        plt.grid()
        plt.plot(trade_pred[i], label=lables[i])
        plt.plot(trade_init[i])
        plt.legend()
            
    plt.show() 
    
def main(TRAIN=True):
    model = MultiLSTM(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, NUM_LAYERS,True)
    merged = tf.summary.merge_all()
    #writer = tf.summary.FileWriter("logs", sess.graph)
    #writer = tf.summary.FileWriter("logs", tf.get_default_graph())
    init = tf.global_variables_initializer()    
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        if TRAIN:
            train(model,sess,saver,merged,init) 
            test(model,sess,saver)
        else:              
            test(model,sess,saver,True)
        
        
if __name__ == '__main__':
    main() # 直接训练
    #main(False) # 跳过训练，直接测试
