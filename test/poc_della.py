import scipy
import torch

delta = torch.rand(5, 3) - 0.5

p = 0.3
eps = -0.15

print(delta)

#Too slow: https://stackoverflow.com/questions/49453455/numpy-argsort-vs-scipy-stats-rankdata
#prefer round up
too_slow = torch.from_numpy(scipy.stats.rankdata(delta.abs(), method="max").reshape(delta.shape))
#print(too_slow)

delta2 = delta.abs().ravel()
#print(delta2)

#https://stephantul.github.io/python/pytorch/2020/09/18/fast_topk/
def get_rank(x, indices):
   vals = x[range(len(x)), indices]
   return (x > vals[:, None]).long().sum(1)

sort_index = torch.argsort(delta2)

# This throws error.
#rank_per_element = get_rank(delta2, sort_index).reshape(delta.shape)

# Use double argsort approach.
rank_per_element = torch.argsort(sort_index).reshape(delta.shape) + 1
print(rank_per_element)

assert torch.allclose(too_slow, rank_per_element, atol = 0.0001)

#We expect the final prob should be [0.63, 0.77] with mean [0.7]
#rank, "center the window"
ne = delta.numel()
to_dare = torch.full(delta.shape, 1 - p) + (rank_per_element / ne - ((ne + 1) / (ne * 2))) * eps 

print(to_dare)

# However in extreme case, they may not be equal.
# Scipy will be all 0 and 1 because of tie breaker.
if True:
   # https://pytorch.org/docs/stable/generated/torch.sort.html
   x = torch.tensor([0, 1] * 9)
   x2 = torch.argsort(torch.argsort(x)) + 1
   x3 = torch.from_numpy(scipy.stats.rankdata(x, method='ordinal'))
   assert torch.allclose(x2, x3, atol = 0.0001)