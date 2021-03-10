# -*- coding: utf-8 -*-
'''
Created on 2021年3月9日

@author: xiaoyw
'''
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM,TimeDistributed,Dense,Dropout
#from keras.layers import LSTM,TimeDistributed,Dense,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

BATCH_START = 0
TIME_STEPS = 15
BATCH_SIZE = 30
INPUT_SIZE = 25
OUTPUT_SIZE = 4
PRED_SIZE = 15 #预测输出1天序列数据
CELL_SIZE = 128
LR = 0.00005
EPOSE = 1000

def get_train_data():
    df = pd.read_csv('share000547.csv')
    return df

def get_test_data():
    df = pd.read_csv('share000547.csv')
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

class KerasMultiLSTM(object):

    def __init__(self,n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size # LSTM神经单元数      
        self.batch_size = batch_size # 输入batch_size大小
    
    def model(self):
        
        self.model = Sequential() 
        
        #LSTM的输入为 [batch_size, timesteps, features],这里的timesteps为步数，features为维度
        # return_sequences = True: output at all steps. False: output as last step.
        # stateful=True: the final state of batch1 is feed into the initial state of batch2
        '''
        self.model.add(LSTM(units = self.cell_size,  activation='relu', return_sequences = True , stateful=True, 
                            batch_input_shape = (self.batch_size, self.n_steps, self.input_size))
        )
        self.model.add(Dropout(0.2))        
        self.model.add(LSTM(units = self.cell_size, activation='relu', return_sequences = True, stateful=True))
        self.model.add(Dropout(0.2))        
        self.model.add(LSTM(units = self.cell_size, activation='relu', return_sequences = True, stateful=True))
        self.model.add(Dropout(0.2))
        '''        
        # 不固定batch_size，预测时可以以1条记录进行分析
        self.model.add(LSTM(units = self.cell_size,  activation='relu', return_sequences = True , 
                            input_shape = (self.n_steps, self.input_size))
        )
        self.model.add(Dropout(0.2))        
        self.model.add(LSTM(units = self.cell_size, activation='relu', return_sequences = True))
        self.model.add(Dropout(0.2))        
        self.model.add(LSTM(units = self.cell_size, activation='relu', return_sequences = True))
        self.model.add(Dropout(0.2))

        #全连接，输出， add output layer
        self.model.add(TimeDistributed(Dense(self.output_size)))
        self.model.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer='adam')
        self.model.summary()

    
    def train(self,x_train,y_train, epochs):
        history = self.model.fit(x_train, y_train, epochs = epochs, batch_size = self.batch_size).history
        self.model.save("lstm-model2.h5")
        
        return history
 
 
if __name__ == '__main__':   
    df = get_train_data()
    train_x,train_y,z,sc,col_name,df = set_datas(df,True)
    # 训练集需要是batch_size的倍数
    k = len(train_x)%BATCH_SIZE
    train_x,train_y = train_x[k:], train_y[k:]

    model = KerasMultiLSTM(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    model.model()
    history = model.train(train_x,train_y,EPOSE)
    
    plt.plot(history['loss'], linewidth=2, label='Train')
    plt.legend(loc='upper right')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()
    
    # 获取原归一化处理的Scaler
    #df = get_train_data()
    #train_x,train_y,z,sc,col_name,df = set_datas(df,True)
    
    df = get_test_data()
    seq,res,z,sc,col_name,df = set_datas(df,False,sc) 
    seq = seq.reshape(-1,TIME_STEPS,INPUT_SIZE)
    #c0 = seq
    #for i in range(BATCH_SIZE-1):
    #    seq = np.row_stack((seq,c0))

    share_close = df['close0'].values
    share_vol = df['vol0'].values/10000
    share_sh = df['close1'].values
    share_sz = df['close2'].values
    model = load_model('lstm-model2.h5')
    pred = model.predict(seq)
    #p1 = pred[0]

    y=get_pred_data(pred[0].reshape(TIME_STEPS,OUTPUT_SIZE),z,sc) 
    #y=get_pred_data(p1,z,sc) 
    df= pd.DataFrame(y,columns=col_name)   
    df.to_csv('yk2.csv') 
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
    