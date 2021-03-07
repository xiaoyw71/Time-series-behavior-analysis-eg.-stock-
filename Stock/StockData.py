'''
Created on 2020年4月7日

@author: xiaoyw
'''
#获取股票数据
import pandas as pd
import tushare as ts
import pymongo
import random
import json
import matplotlib.pyplot as plt
import numpy as np
import re

class StockHisData(object):

    def __init__(self, pro=True):
        self.pro = pro
        self.columns = ['ts_code','trade_date','open','high','close','low','vol','change','pct_chg']
        if self.pro:
            ts.set_token('3df2b196903a103ca46260061636728625ee5e02b694a031b392ed57')
            self.stock = ts.pro_api()
    #设置股票编码，老版本转换
    def setCodebyOld(self):
        #追加股票代码
        if self.code[0]=='0':
            self.code = self.code + '.SZ'
        elif self.code[0]=='6':
            self.code = self.code + '.SH'
        elif self.code == 'sh':
            # 上证指数
            self.code = '000001' + '.SH'
        elif self.code == '399001':
            #深成指数
            self.code = self.code + '.SZ' 
                               
    #获取历史日线数据    
    def get_his_dat(self,start_date,end_date): 
        #新pro接口，可以多个股票
        if self.pro:        
            self.his_dat = self.stock.daily(ts_code= self.code, start_date=start_date, end_date=end_date)
            #print(self.his_dat)
        else:
            #旧接口，不用注册
            #code：股票代码，即6位数字代码，或者指数代码（sh=上证指数 sz=深圳成指 hs300=沪深300指数 sz50=上证50 zxb=中小板 cyb=创业板）
            self.his_dat = ts.get_hist_data(code=self.code,start=start_date, end=end_date)
            #把索引赋值给trade_date
            #self.his_dat['trade_date'] = self.his_dat.index
            self.his_dat = self.his_dat.reset_index()
            self.setCodebyOld()
                            
            self.his_dat['ts_code'] = self.code
            #参照pro接口，修改列名
            self.his_dat = self.his_dat.rename(columns={'date':'trade_date','volume':'vol','price_change':'change','p_change':'pct_chg'})
        #筛选列            
        self.his_dat = self.his_dat[self.columns] #.reset_index()
        
        return self.his_dat
    #获取美股日线数据
    def get_us_dat(self,start_date,end_date): 
        if self.pro:          
            self.us_dat = self.stock.us_daily(ts_code= self.us_code, start_date=start_date, end_date=end_date)
            print(self.us_dat)
        
        return self.us_dat
    #获取沪深指数
    def get_hs_index(self,start_date,end_date):
        if self.pro:
            start_date=re.sub('\D','',start_date)    
            end_date = re.sub('\D','',end_date) 
            self.hs_index = ts.pro_bar(ts_code= self.code, asset='I', start_date=start_date, end_date=end_date)
            print(self.hs_index)
        else:
            #旧接口，不用注册
            index_code={'000001.SH':'sh','399001.SZ':'399001','000300.SH':'000016.SH','sz50':'sz50','399005.SZ':'zxb','399006.SZ':'cyb'}
            
            self.his_dat = ts.get_hist_data(code=index_code[self.code],start=start_date, end=end_date)
            #把索引赋值给trade_date
            #self.his_dat['trade_date'] = self.his_dat.index
            self.his_dat = self.his_dat.reset_index()
                            
            self.his_dat['ts_code'] = self.code
            #参照pro接口，修改列名
            self.his_dat = self.his_dat.rename(columns={'date':'trade_date','volume':'vol','price_change':'change','p_change':'pct_chg'})
        #筛选列            
        self.his_dat = self.his_dat[self.columns] #.reset_index()           
        
        return self.hs_index     
    # 获取复权数据
    def get_h_dat(self,start_date,end_date,fq='hfq'):
        #self.h_dat = ts.get_h_data(code=self.code, autype='hfq',start=start_date, end=end_date)
        self.h_dat = ts.pro_bar(ts_code=self.code, adj=fq, start_date=start_date, end_date=end_date)
                                                                         
        return self.h_dat
    
    #获取指数
    def get_hs_index_pro(self,start_date,end_date):
        self.hs_index = self.pro.index_daily(ts_code=self.code, start_date=start_date, end_date=end_date)
        return self.hs_index
    
    #美股指数
    def get_us_index(self,start_date,end_date):
        if self.pro:          
            self.us_index = self.stock.index_global(ts_code= self.us_code, start_date=start_date, end_date=end_date)
            print(self.us_index)
            self.us_index = self.us_index[self.columns]
        
        return self.us_index    
        
    def set_code(self,code):
        self.code = code
        
    def set_us_code(self,code):
        self.us_code = code    
    #获取分钟级别数据    
    def get_tickshare_dat(self,freq,start_date, end_date):
        if self.pro: 
            start_date=re.sub('\D','',start_date)    
            end_date = re.sub('\D','',end_date) 
            freq = freq + 'min'    
            self.tickshare_dat = ts.pro_bar(ts_code=self.code, freq = freq,start_date=start_date, end_date=end_date)
            self.tickshare_dat['vol'] = self.tickshare_dat['vol'] /100
        else:
            # ktype：数据类型，D=日k线 W=周 M=月 5=5分钟 15=15分钟 30=30分钟 60=60分钟，默认为D
            self.tickshare_dat = ts.get_hist_data(code=self.code, ktype = freq,start=start_date, end=end_date)
            self.tickshare_dat['ts_code'] = self.code
            self.tickshare_dat = self.tickshare_dat.reset_index()
            self.tickshare_dat = self.tickshare_dat.rename(columns={'date':'trade_time','volume':'vol'})
            self.tickshare_dat['trade_date'] = self.tickshare_dat['trade_time'].apply(lambda x:re.sub('\D','',x[0:10]))
            self.setCodebyOld()                            
            self.tickshare_dat['ts_code'] = self.code
        self.tickshare_dat = self.tickshare_dat[['ts_code','trade_time','open','high','close','low','vol','trade_date']]                   
            
        return self.tickshare_dat
    #获取股票基本面信息
    def get_ShareInfo(self,trade_date):
        if self.pro:
            self.shareInfo = self.stock.daily_basic(ts_code=self.code, trade_date=trade_date) #, fields='ts_code,trade_date,turnover_rate,volume_ratio,pe,pb')
        else:
            self.shareInfo = ts.get_stock_basics()
            
        print(self.shareInfo)
    

class Stock_Collection(object):
    def __init__(self,db_name):
        self.db_name = db_name
        client = pymongo.MongoClient('mongodb://stock:stock@localhost:27017/stock')
        self.db = client[self.db_name]
        
    def insertdatas(self,name,datas):
        collection = self.db[name]   
        
        collection.insert(json.loads(datas.T.to_json()).values())
        
    def getDistinctCode(self,name):
        collection = self.db[name]
        
        code = collection.distinct('ts_code')   
        
        return code
    
    def setIndex_Code(self):
        self.sentiment_index = ['IXIC','DJI','HSI'] # 情绪指数
        self.sentiment_index_column = ['trade_date','open','high','close','low','change','pct_chg']
        self.index_daily = ['000001.SH', '399001.SZ']
        self.index_daily_column = ['trade_date','open','high','close','low','vol','change','pct_chg']
        
    def setCode(self):
        self.code = ['002230.SZ'] #, '000547.SZ', '601318.SH', '601208.SH', '600030.SH', '000938.SZ', '002108.SZ', '600967.SH']
        self.stock_column = ['trade_date','open','high','close','low','vol','change','pct_chg']
    # 构造LSTM模型训练集    
    def generate_train_datas(self,db_name,code_name):
        collection = self.db[db_name]
        self.out_code = code_name
        #查询条件“字典”
        query_dict = {'ts_code':'1'}
        #col_name = {'_id':0,'trade_date':1,'ts_code':1,'open':1,'high':1,'close':1,'low':1,'vol':1,'change':1,'pct_chg':1}
        col_name = {'_id':0}
        for d in self.stock_column:
            col_name[d] = 1

        query_dict['ts_code'] = self.out_code
        #注意时间排序
        df = pd.DataFrame(list(collection.find(query_dict,col_name).sort([('trade_date',1)])))
        df['trade_date'] = df['trade_date'].apply(lambda x:re.sub('\D','',x)) #去掉日期中的“-”符号
        self.code.remove(self.out_code)  # 删除输出股票代码
        #构造股票数据集
        n = 0
        k = 0
        columns = self.stock_column.copy()
        columns.remove('trade_date') 
        print('Start!')
        #self.code长度为1，下面循环不执行
        for code in self.code:
            query_dict['ts_code'] = code
            df1 = pd.DataFrame(list(collection.find(query_dict,col_name).sort([('trade_date',1)])))
            df1['trade_date'] = df1['trade_date'].apply(lambda x:re.sub('\D','',x)) #去掉日期中的“-”符号
            #按日期合并两个表
            #df =pd.merge(left=df,right=df1,how='left',on=['trade_date'])
            #以上证为基准
            df =pd.merge(left=df,right=df1,how='inner',on=['trade_date'])
            # 处理合并表，字段重复的情况，需要把_x,_y新命名字段，下轮继续
            cols_dict = {}
            for cols in columns:
                cols_dict[cols+'_x'] = cols + str(n)
                cols_dict[cols+'_y'] = cols + str(n+1)             
            if k==0:
                df = df.rename(columns=cols_dict)    
                n = n + 2
                k = 1
            else:
                k = 0
            print('code 1')
            print(df)
        #构造数据集——上证、深成指数
        columns = self.index_daily_column.copy() #默认list为传址，需要赋值新list
        columns.remove('trade_date')   
        print(self.index_daily_column)      
        for index_daily in self.index_daily:
            query_dict['ts_code'] = index_daily
            col_name = {'_id':0}
            for d in self.index_daily_column:
                col_name[d] = 1
            df1 = pd.DataFrame(list(collection.find(query_dict,col_name).sort([('trade_date',1)])))
            df1['trade_date'] = df1['trade_date'].apply(lambda x:re.sub('\D','',x)) #去掉日期中的“-”符号
            #按日期合并两个表
            df =pd.merge(left=df,right=df1,how='left',on=['trade_date'])
            cols_dict = {}
            for cols in columns:
                cols_dict[cols+'_x'] = cols + str(n)
                cols_dict[cols+'_y'] = cols + str(n+1)
            if k==0:
                df = df.rename(columns=cols_dict)    
                n = n + 2
                k = 1
            else:
                k = 0
         
            print(df)
        #构造数据集——情绪指数
        columns = self.sentiment_index_column.copy()
        columns.remove('trade_date')              
        for sentiment_index in self.sentiment_index:
            query_dict['ts_code'] = sentiment_index
            col_name = {'_id':0}
            for d in self.sentiment_index_column:
                col_name[d] = 1
            df1 = pd.DataFrame(list(collection.find(query_dict,col_name).sort([('trade_date',1)])))
            df1['trade_date'] = df1['trade_date'].apply(lambda x:re.sub('\D','',x)) #去掉日期中的“-”符号
            #按日期合并两个表
            df =pd.merge(left=df,right=df1,how='left',on=['trade_date'])
            cols_dict = {}
            for cols in columns:
                cols_dict[cols+'_x'] = cols + str(n)
                cols_dict[cols+'_y'] = cols + str(n+1)
            df = df.rename(columns=cols_dict)    
            if k==0:
                df = df.rename(columns=cols_dict)    
                n = n + 2
                k = 1
            else:
                k = 0              
        print(df)
        df = df.fillna(0) #数据缺失补上为0，相当于停盘！！！
        df.to_csv('share20210306.csv')


def test_pro_us_index():
    SC = Stock_Collection('stock')
    us_index_code='IXIC,DJI,HSI,SPX,N225,GDAXI'
    share = StockHisData(True)
    share.set_us_code(us_index_code)

    df = share.get_us_index('20210301', '20210306')
    SC.insertdatas('fq_stocks', df)

    print(df)
 
def add_dat():
    SC = Stock_Collection('stock')
    df = pd.read_csv('000001.csv')
    df['ts_code'] = '000001.SH'
    print(df)   
    SC.insertdatas('fq_stocks', df) 
    df = pd.read_csv('399001.csv')
    df['ts_code'] = '399001.SZ'
    print(df) 
    SC.insertdatas('fq_stocks', df)   
 
        

def test_pro():
    share = StockHisData(True)
    #share.set_code('000001.SH')
    share.set_code('601318.SH')
    print('Share Code is %s'%(share.code))

    df = share.get_his_dat('20210101', '20210210')

    print(df)

def test_pro_index():
    SC = Stock_Collection('stock')  
    share = StockHisData(True)
    #share.set_code('000001.SH')
    share.set_code('000001.SH')
    print('Share Code is %s'%(share.code))

    df = share.get_hs_index_pro('20210301', '20210306')
    SC.insertdatas('fq_stocks', df) 

    share.set_code('399001.SZ')
    print('Share Code is %s'%(share.code))

    df = share.get_hs_index_pro('20210301', '20210306')
    SC.insertdatas('fq_stocks', df) 

    print(df)        

def test_old_index():
    SC = Stock_Collection('stock')    
    share = StockHisData(False)
    # 上证指数
    share.set_code('sh')
    df = share.get_his_dat('2021-03-01', '2021-03-06')
    SC.insertdatas('fq_stocks', df) 
    
    #深成指数
    share.set_code('399001')
    df = share.get_his_dat('2021-03-01', '2021-03-06')
    SC.insertdatas('fq_stocks', df) 
    


def test_old():
    SC = Stock_Collection('stock')
    share = StockHisData(False)       

    #df = pd.DataFrame()
    share_code = ['002230','000547','601318','601208','600030','000938','002108','600967']
    for i in range(len(share_code)):
        share.set_code(share_code[i])
        df = share.get_his_dat('2018-01-01', "2021-02-19") 
        SC.insertdatas('stocks', df) 

        print(df)      

def test_fq():
    SC = Stock_Collection('stock')
    share = StockHisData(False)       

    #df = pd.DataFrame()
    share_code = ['002230.SZ','000547.SZ','601208.SH','600030.SH','601318.SH']
    #share_code = ['399001.SZ','000001.SH']
    for i in range(len(share_code)):
        share.set_code(share_code[i])
        df = share.get_h_dat('2021-03-01', "2021-03-06" ,fq='hfq') 
        #df = share.get_h_dat('2021-03-01', "2021-03-06" ,fq='None') 
        SC.insertdatas('fq_stocks', df) 

        print(df)    

def test_old_tickshare():
    #SC = Stock_Collection('stock')
    share = StockHisData(False)
    share_code = '002230'
    share.set_code(share_code)
    df = share.get_tickshare_dat('15', '2018-01-01', "2021-02-19")
     
    print(df)     

if __name__ == '__main__':
    #test_old()
    #test_old_index()
    #test_old_tickshare()
    #test_pro_index()
    #test_pro_index()
    #美国道琼斯、纳斯达克等指数
    #test_pro_us_index()
    #test_fq()
    
    SC = Stock_Collection('stock')
    #print(SC.getDistinctCode('stocks'))
    SC.setIndex_Code()
    SC.setCode()
    #复权
    SC.generate_train_datas('fq_stocks', '002230.SZ')
    
    #share = StockHisData(True)
    #share.set_code('002230.SZ')
    #share.get_ShareInfo('20210219')
    
    



           