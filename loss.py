import torch


def cosine_sim(zi, zj, eps=1e-8, temp=0.5):
    a_n, b_n = zi.norm(dim=1)[:, None], zj.norm(dim=1)[:, None]
    a_norm = zi / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = zj / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1)) / temp
    return sim_mt

def nt_xent(s):
    l = -torch.log(torch.exp(s) / (torch.sum(torch.exp(s), 0) - torch.exp(torch.diag(s))))
    return l

def cus_loss(u, v, temp=0.5):
    N = u.shape[0]
    Z = torch.cat((u.unsqueeze(-1), v.unsqueeze(-1)), 1).reshape(len(u) + len(v), -1)
    s = cosine_sim(Z, Z, temp=temp)
    l = nt_xent(s)
    loss = 0
    for i in range(N):
        loss += l[i][i+N] + l[N+i][i]
        loss /= (2*N)
    return loss

def loop_loss(u,v):
    Z = []
    for i in range(N):
        Z.append(u[i])
        Z.append(v[i])
        Z = torch.stack(Z) 

        S = torch.randn(2*N, 2*N)
        for i in range(2*N):
            for j in range(2*N):
                S[i][j] = cosine_sim(Z[i].reshape(1, -1), Z[j].reshape(1, -1))

        L = torch.randn(2*N, 2*N)
        a = []
        for i in range(2*N):
            for j in range(2*N):
                denom = torch.sum(torch.exp(S[i])) - torch.exp(S[i][i])
                a.append(denom)
                L[i][j] = -torch.log(torch.exp(S[i][j]) / denom)

        loss = 0
            # loss = torch.sum(torch.tensor([l[i][i+N] + l[N+i][i] for i in range(N)]) / (2*N))
        for i in range(N):
            loss += L[i][i+N] + L[N+i][i]
            loss /= (2*N)
        return loss