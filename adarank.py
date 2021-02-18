import torch

y = qrels


class WeakRanker:
    def __init__(self, P, x, y):
        pass

    def forward(x):
        # return scores of x
        pass

    def sort(x):
        # evaluate all docs and sort by score
        pass


class BoostedRanker:
    def __init__(self, P, x, y):
        torch.sum(alphas * hs(x))
        pass

    def forward(x):
        # return scores of x
        pass

    def sort(x):
        # evaluate all docs and sort by score
        pass


def metric(pi, y):
    # find a listwise metric to optimize
    pass


def loss():
    return torch.sum(torch.log(1 + torch.exp(-metric(pi, y))))  # exponential loss


# load qrels of top100?

# figure out what feature vectors to use

P = 1 / len(dataset) * torch.ones(len(dataset))

alphas = []
hs = []

for t in range(num_iters):
    x, y = next(dataloader)

    h = WeakRanker(P, x, y)
    hs.append(h)

    pi = h.rank(x)

    alpha = 1 / 2 * torch.log((torch.sum(P * (1 + metric(pi, y)))) / (P * (1 - metric(pi, y))))
    alphas.append(alpha)

    f = BoostedRanker(alphas, hs)

    pi = f.rank(x)

    P = torch.nn.functional.softmax(metric(pi, y))

