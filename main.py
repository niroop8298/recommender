from fastai.collab import *
from fastai.tabular.all import *
import re
from functools import reduce
from datetime import date
data=pd.read_csv('blore_daily.csv')
#data=data[data['delivery_slot']=='express']
data=data[data['delivery_slot']=='scheduled']
data=data.rename(columns={'customer_id':'user_id','dt':'created_date'})
import matplotlib.pyplot as plt
def based_on_count():
    
    def cleanup1(y):
        if '(' in y:
            a,b=y.split('(',1)
            return a.split('/')[0]+' ('+b
        else:
            return y
    def cleanup2(y):
        for i in ['Premium','Fresh','Yummy','Antibiotic-residue-free','Meaty','Tender','Cleaned']:
            y=y.replace(i,'')
        y=y.strip()
        return y
    def cleanup3(y):
        pattern = r"\([^()]*?(?<!kg)(?<!g)\)"
        result = re.sub(pattern, "",y)
        return result
        
    data['item_name']=data['item_name'].apply(cleanup1)
    data['item_name']=data['item_name'].apply(cleanup2)
    data['item_name']=data['item_name'].apply(cleanup3)
    
    
    x=data.groupby(['user_id','product_id'])['item_id'].count()
    df = x.reset_index()
    df.columns = ['user_id','product_id','count']
    df=df[df['count']>=5]
    df['count']=np.log10(df['count']*9+1)
    df['count']=(df['count']-df['count'].min())/(df['count'].max()-df['count'].min())
    
    df['count']=df['count']*(5/df['count'].max())
   
    return df

def based_on_weight():
    def cleanup1(y):
        if '(' in y:
            a,b=y.split('(',1)
            return a.split('/')[0]+' ('+b
        else:
            return y
    def cleanup2(y):
        for i in ['Premium','Fresh','Yummy','Antibiotic-residue-free','Meaty','Tender','Cleaned']:
            y=y.replace(i,'')
        y=y.strip()
        return y
    def cleanup3(y):
        pattern = r"\([^()]*?(?<!kg)(?<!g)\)"
        result = re.sub(pattern, "",y)
        return result
        
    data['item_name']=data['item_name'].apply(cleanup1)
    data['item_name']=data['item_name'].apply(cleanup2)
    data['item_name']=data['item_name'].apply(cleanup3)
    x=data.groupby(['user_id','product_id'])['net_weight'].sum()
    df = x.reset_index()
    df.columns = ['user_id','product_id','count']
    df=df[df['count']>1.2]

    df['count']=np.log10(df['count']*9+1)
    df['count']=(df['count']-df['count'].min())/(df['count'].max()-df['count'].min())
    #df['count']=df['count']*(5/df['count'].max())
    print(df['count'])
    #df['count'].plot(kind='kde')
    return df

def based_on_datetime():
    def cleanup1(y):
        if '(' in y:
            a,b=y.split('(',1)
            return a.split('/')[0]+' ('+b
        else:
            return y
    def cleanup2(y):
        for i in ['Premium','Fresh','Yummy','Antibiotic-residue-free','Meaty','Tender','Cleaned']:
            y=y.replace(i,'')
        y=y.strip()
        return y
    def cleanup3(y):
        pattern = r"\([^()]*?(?<!kg)(?<!g)\)"
        result = re.sub(pattern, "",y)
        return result
    def func5(user,product):
        y=data.loc[(data['user_id']==user) & (data['product_id']==product)]['created_date']
        L=[]
        for i in y:
            L.append(i.split()[0])
        if len(L)>1:
            year,month,day=map(int,L[0].split('-'))
            d0 = date(year,month,day)
            total=0
            for i in L:
                year,month,day=map(int,i.split('-'))
                d1=date(year,month,day)
                if d1==d0:
                    continue
                total=total+(90-((d1-d0).days))**2
                d0=d1
         
            return total
        else:
            return 0
        
    data['item_name']=data['item_name'].apply(cleanup1)
    data['item_name']=data['item_name'].apply(cleanup2)
    data['item_name']=data['item_name'].apply(cleanup3)
    
   
    
    #x=data.groupby(['user_id','product_id'])
    x=data.groupby(['user_id','product_id'])['item_id'].count()
    df = x.reset_index()
    df.columns = ['user_id','product_id','count']
    #df=df.iloc[:1000]
    df['count'] = df.apply(lambda x: func5(x.user_id,x.product_id), axis=1)
    df=df[df['count']>0]
    df['count']=np.log10(df['count']*9+1)
    df['count']=(df['count']-df['count'].min())/(df['count'].max()-df['count'].min())
    df['count']=df['count']*(5/df['count'].max())
    return df
    #func5()
    #df['count']=df['count'].apply(func5)
    
def based_on_datetime2():
    def cleanup1(y):
        if '(' in y:
            a,b=y.split('(',1)
            return a.split('/')[0]+' ('+b
        else:
            return y
    def cleanup2(y):
        for i in ['Premium','Fresh','Yummy','Antibiotic-residue-free','Meaty','Tender','Cleaned']:
            y=y.replace(i,'')
        y=y.strip()
        return y
    def cleanup3(y):
        pattern = r"\([^()]*?(?<!kg)(?<!g)\)"
        result = re.sub(pattern, "",y)
        return result
    def func5(user,product):
        y=data.loc[(data['user_id']==user) & (data['product_id']==product)]['created_date']
        L=[]
        first_date=data.loc[0]['created_date']
        d0=date(first_date.year,first_date.month,first_date.day)
        total=0
        for i in y:
            d1=date(i.year,i.month,i.day)
            total=total+(d1-d0).days
        return total
            
    data['item_name']=data['item_name'].apply(cleanup1)
    data['item_name']=data['item_name'].apply(cleanup2)
    data['item_name']=data['item_name'].apply(cleanup3)
    data['created_date']=pd.to_datetime(data['created_date'])
    data.sort_values(by='created_date')
    x=data.groupby(['user_id','product_id'])['item_id'].count()
    df = x.reset_index()
    df.columns = ['user_id','product_id','count']
    df['count'] = df.apply(lambda x: func5(x.user_id,x.product_id), axis=1)
    df=df[df['count']>50]
    df['count']=np.log10(df['count']*9+1)
    df['count']=(df['count']-df['count'].min())/(df['count'].max()-df['count'].min())
    df['count']=df['count']*(5/df['count'].max())
    return df
df=based_on_datetime2()
dls = CollabDataLoaders.from_df(df,item_name='product_id', bs=64)
class DotProductBias(Module):
    def __init__(self, n_users, n_products, n_factors, y_range=(0,5.5)):
        self.user_factors = Embedding(n_users, n_factors)
        self.user_bias = Embedding(n_users, 1)
        self.product_factors = Embedding(n_products, n_factors)
        self.product_bias = Embedding(n_products, 1)
        self.y_range = y_range
        
    def forward(self, x):
        users = self.user_factors(x[:,0])
        products = self.product_factors(x[:,1])
        res = (users * products).sum(dim=1, keepdim=True)
        res += self.user_bias(x[:,0]) + self.product_bias(x[:,1])
        return sigmoid_range(res, *self.y_range)
n_users  = len(dls.classes['user_id'])
n_products = len(dls.classes['product_id'])
model = DotProductBias(n_users, n_products, 50)
print(n_users,n_products,len(model.product_factors.weight))
learn = Learner(dls, model, loss_func=MSELossFlat())
#learn.lr_find()
learn.fit_one_cycle(5, 4e-2, wd=0.1)
ratings=df
#learn=load_learner('blr_model_scheduled.pkl')
g = ratings.groupby('product_id')['count'].count()
top_products = g.sort_values(ascending=False).index.values[:1000]

#top_products=ratings.groupby('product_id').sum()
#top_products = top_products.sort_values(by='count',ascending=False).index.values[:1000]
top_idxs = tensor([learn.dls.classes['product_id'].o2i[m] for m in top_products])
product_w = learn.model.product_factors.weight[top_idxs].cpu().detach()
product_pca = product_w.pca(3)
fac0,fac1,fac2 = product_pca.t()
idxs = list(range(50))
X = fac0[idxs]
Y = fac2[idxs]
plt.figure(figsize=(12,12))
plt.scatter(X, Y)
def get_item_name(y):
    return data.loc[data['product_id']==y].iloc[0]['item_name']

for i, x, y in zip(top_products[idxs], X, Y):
 
    plt.text(x,y,get_item_name(i), color=np.random.rand(3)*0.7, fontsize=11)
plt.show()
