# TFJ-DRL-Replication
A paper replication project for *Time-driven feature-aware jointly deep reinforcement learning (TFJ-DRL) for financial signal representation and algorithmic trading*. 

Algorithmic trading has become a hot topic since the adoption of computers in stock exchanges. There are two categories of algorithmic trading, one based on prior knowledge and another based on machine learning (ML). The latter is gaining more attentions these days, as comparing to methods based on prior knowledge, ML based methods does not require professional financial knowledge, research, or trading experience. 

However, there are several drawbacks to previous implementations of machine learning algorithmic trading:  

* Supervised learning methods are difficult to achieve online learning, due to the cost of training. They attempt to predict stock prices of the next time point, but accuracy of price prediction results in second error propagation during translation from price prediction to trading actions.

* Reinforcement learning (RL) methods lacks the ability to perceive and represent environment features, as well as the ability to dynamically consider past states and changing trends. 

The paper of interest (TFJ-DRL) aims to combine the strength from both deep learning and reinforcement learning by integrating Recurrent Neural Network (RNN) and policy gradient RL.

Through this document, I'm going to detail the steps I took and decisions I made to replicate the model TFJ-DRL. 

## Content Overview

This document is divided into 9 parts:

0. Developement Environment
1. Data used in model and data acquisition
2. Data preprocessing
3. RNN model definition
4. Reinforcement model definition
5. Loss function design
6. Model training and weight selection
7. Model performance testing
8. Link to paper and other resources

## Development Environment

The python packages used in the project include: PyTorch, TorchVision, Numpy, Pandas, MatplotLib, YFinance, Statsmodels and TA-Lib.

YFinance, Statsmodels, and TA-Lib can be installed via:
```
conda install -c anaconda statsmodels
conda install -c ranaroussi yfinance
conda install -c quantopian ta-lib
```

On Google Colab, TA-Lib needs to be seperated installed via:
```
!wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
!tar -xzvf ta-lib-0.4.0-src.tar.gz
%cd ta-lib
!./configure --prefix=/usr
!make
!make install
!pip install Ta-Lib
```

## Data used in model and data acquisition

Data used in the projects are daily stock trading history that can be downloaded from Yahoo Finance. The historical stock data used in the project is from Jan 2013 to Dec 2018.

A sample data snippet looks like:

| Time | Open | Close | High | Low | Volume
| ------- | ------- | ------- | ------- | ------- | ------- |
| 2018/01/02 | 683.73 | 695.89 | 682.36 | 695.81 | 7250000 |
| 2018/01/03 | 697.85 | 714.37 | 697.77 | 713.06 | 8660000 |
| ... | ... | ... | ... | ... | ... |

For each stock, given its ticker and the start/end time we are interested in, we can get the required data via:
```
#read daily stock data series from yahoo finance
def get_data(name, start="2017-01-01", end="2020-01-01"):
    ticker = yf.Ticker(name)
    data=ticker.history(start=start,  end=end)
    return data.drop(['Dividends', 'Stock Splits'], axis=1, errors='ignore')
```

Here, we delete *Dividents* and *Stock Splits* columns because they are not required in the model. 

## Data preprocessing

Next, given the full stock data, we will use TA-Lib to calculate technical analysis indicators for each day:
```
def calc_tech_ind(data):
    #overlap 
    data['upbd'], data['midbd'], data['lowbd'] = ta.BBANDS(data["Close"])
    data['dema'] = ta.DEMA(data["Close"], timeperiod=30)
    data['tema'] = ta.TEMA(data["Close"], timeperiod=30)
    ...

    return data
```

In total, 40 features are calculated from Overlap Studies, Momentum Indicators, Volume Indicators, Volitility Indicators, and Cycle Indicators. 

Together with the original 5 features (*Time* is not a feature), there are 45 features for each stock. 

However, for this model, to complete features for stock **X**, the stock we are interested in, we will also consider features from stocks with similar trends with **X**. 

To do this, we need to first have a predetermined list of stocks to consider, and calculate features for each of them:
```
def get_data_set(stock_id, name_list, start="2017-01-01", end="2020-01-01"):
    data_list=[]
    for name in name_list:
        data_list.append(calc_tech_ind(get_data(name, start, end)).iloc[90:].fillna(0).values)
    ...
```

Next up, we need to compute the correlation between X and each stock from the list, and it can be computed in Cointegration test. Cointegration test outputs a value between [0, 1], with smaller values indicate larger correlations. The *high_correlation_list* is a list containing the index of stocks with similar trends as **X**. 
```
def get_data_set(stock_id, name_list, start="2017-01-01", end="2020-01-01"):
    ...
        
    #get number of original
    feature_count=data_list[0].shape[1]
    #calculate cointegration
    high_correlation_list=[]
    for j in range(len(data_list)):
        if stock_id != j:
            coint=ts.coint(data_list[stock_id][:, 3], data_list[j][:, 3])[1] 
            if coint <= 0.1:
                high_correlation_list.append(j)
            
    return data_list, high_correlation_list
```

To make *Back Propagation Through Time* easier for training, we need to slice our long dataset into shorter pieces. But first, we will need to concatenate features from stocks with high correlation trends:
```
def toSequential(stock_id, name_list, timeStep=24, gap=12, start="2017-01-01", 
                 end="2020-01-01", use_external_list=False, external_list=[]):
    data_list, hcl=get_data_set(stock_id, name_list, start=start, end=end) 
    #For Validation and Testing, keep using the same HCL list. 
    if (use_external_list):
      hcl=external_list
      
    #append coint features to the end
    avg_features=np.zeros((data_list[stock_id].shape[0], data_list[stock_id].shape[1]-4))
    for k in hcl:
        feature=data_list[k][:, 4:]
        avg_features+=(feature-feature.mean(axis=0, keepdims=True))/(feature.std(axis=0, keepdims=True))
    #append to the end
    stkData=np.concatenate([data_list[stock_id], avg_features], axis=1)

```

In addition to *X*(normalized stock features of t), *y*(normalized close price of t+1) we will have *z*(normalized price difference between t+1 and t) and *zp*(un-normalized price difference between t+1 and t). As *y* and *z* will be used during training and *z* and *zp* will be used during evaluation and testing. 

```
#generate x, y, z, zp quadruples to sequence according to $timeStep and $gap
#x: historical data w/ technical analysis indicator
#y: closing price of t+1
#z:  difference between t+1 and t step's closing price
#zp:  difference between t+1 and t step's closing price, un-normalized
#hcl: high correlation list from get_data_set function
def toSequential(stock_id, name_list, timeStep=24, gap=12, start="2017-01-01", 
                 end="2020-01-01", use_external_list=False, external_list=[]):
	...

    #closing: from id=0 to last
    closing=stkData[:, 3]
    #data from id=0 to second to last
    data=stkData[:-1]
    #calculating number of available sequential samples
    data_length=len(data)
    count=(data_length-timeStep)//gap+1
    stockSeq=[]
    labelSeq=[]
    diffSeq=[]
    realDiffSeq=[]
    for i in range(count):
        #segData dims: [timestep, feature count]
        segData=data[gap*i:gap*i+timeStep]
        segClosing=closing[gap*i:gap*i+timeStep+1]
        #segDiff=diff[gap*i:gap*i+timeStep]
        #normalization
        segDataNorm=np.nan_to_num((segData-segData.mean(axis=0, keepdims=True))/segData.std(axis=0, keepdims=True))
        segClosingNorm=(segClosing-segClosing.mean())/segClosing.std()
        #segDiff=(segDiff-segDiff.mean())/segDiff.std()
        
        stockSeq.append(segDataNorm)
        labelSeq.append(segClosingNorm[1:])
        diffSeq.append(segClosingNorm[1:]-segClosingNorm[:-1])
        realDiffSeq.append(segClosing[1:]-segClosing[:-1])
    stockSeq=np.array(stockSeq)
    labelSeq=np.array(labelSeq)
    diffSeq=np.array(diffSeq)
    realDiffSeq=np.array(realDiffSeq)
    return (stockSeq.astype('float32'), labelSeq.astype('float32'),
    diffSeq.astype('float32'), realDiffSeq.astype('float32'), hcl)

```

The final step is run-of-the-mill dataset definition:
```
#input each step:  vector including [stock info, tech indicators]
#output each step: closing price t+1, price diff between t+1 and t
#full_list: output from get_data_set

class StockDataset(Dataset):
    def __init__(self, stock_id, name_list, transform=None, timestep=24, gap=12,
                 start="2017-01-01", end="2020-01-01",
                 use_external_list=False, external_list=[]):
        self.transform=transform
        self.id=stock_id
        
        
        #load data into cohort
        X, y, z, zp, hcl=toSequential(stock_id, name_list, timeStep=timestep, 
                                      gap=gap, start=start, end=end, 
                                      use_external_list=use_external_list, 
                                      external_list=external_list)

        self.X=X
        self.y=y  
        self.z=z  
        self.zp=zp
        self.high_correlation_list=hcl
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        """
        data returned in the format of 
        """
        if torch.is_tensor(idx):
            idx=idx.tolist()
        
        data=self.X[idx]
        label1=self.y[idx]
        label2=self.z[idx]
        label3=self.zp[idx]
        if self.transform:
            data=self.transform(data)
        return (data, label1, label2, label3)
    
    def getHighCorrelationList(self):
        return self.high_correlation_list

    def getDS(self):
        return self.X, self.y, self.z, self.zp
```



