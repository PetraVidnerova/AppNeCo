import torch
import torch.nn as nn

class SparseLinear(nn.Module):
    def __init__(self, w, b):
        super().__init__()
        self.w = w
        self.b = b
        if self.b is not None:
            self.b = b.flatten()

        if self.b is not None:
            assert self.w.shape[0] == self.b.shape[0]    

    def forward(self, x):
        device = x.device
        self.w = self.w.to(device)
        if self.b is not None:
            self.b = self.b.to(device)
        return nn.functional.linear(x.flatten(), self.w, self.b)


def left_padding(x, padding):
    moved_indices = torch.vstack([
        x._indices()[0],
        x._indices()[1] + padding
    ])

    return torch.sparse_coo_tensor(
        indices = moved_indices,
        values = x._values(),
        size = (x.shape[0], x.shape[1]+padding)
    )


def right_padding(x, padding):
    return torch.sparse_coo_tensor(
        indices = x._indices(),
        values = x._values(),
        size = (x.shape[0], x.shape[1]+padding)
    )


def multidot(list_W):

    if all([W.is_sparse for W in list_W]):
        i = 0 
        result = list_W[i].cpu()
        i += 1
        while i < len(list_W):
            w = list_W[i].cpu()
            result = torch.sparse.mm(result, w)
            i += 1
        return result.cuda()
        
    return torch.linalg.multi_dot(list_W)
