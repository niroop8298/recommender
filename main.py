from fastai.collab import *
from fastai.tabular.all import *
class DotProduct(Module):
    def __init__(self, n_users, n_movies, n_factors):
        self.user_factors = Embedding(n_users, n_factors)
        self.movie_factors = Embedding(n_movies, n_factors)
    def forward(self, x):
        users = self.user_factors(x[:,0])
        movies = self.movie_factors(x[:,1])
        return (users * movies).sum(dim=1)
path = untar_data(URLs.ML_100k)
data=pd.read_excel('Book1.xlsx')
data=pd.DataFrame(data)
dls = CollabDataLoaders.from_df(data, item_name='movieId',bs=15)
n_users  = len(dls.classes['userId'])
n_movies = len(dls.classes['movieId'])
n_factors = 5
model = DotProduct(n_users, n_movies, 5)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(53, 2e-1)
movie_factors = learn.model.movie_factors.weight
user_factors=learn.model.user_factors.weight
movieId=27
userId=563
idx=dls.classes['movieId'].o2i[movieId]
idx2=dls.classes['userId'].o2i[userId]
val=torch.matmul(movie_factors[idx],user_factors[idx2].t())
print("Predicted rating is",val)
