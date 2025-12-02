import torch
import torch.nn as nn
import torch.distributed as dist


class GatherLayer(torch.autograd.Function):
    """Gather layer for distributed SimCLR training"""
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[torch.distributed.get_rank()]
        return grad_out


class AllGather(torch.autograd.Function):
    """Gather layer for distributed PAWS training"""
    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduce(torch.autograd.Function):
    """Reduce layer for distributed PAWS training"""
    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class NTXent(nn.Module):
    """Normalized temperature-scaled cross-entropy loss from https://github.com/Spijkervet/SimCLR"""
    def __init__(self, tau=1, actual_batch_size=None, gather_grads=False, world_size=1):
        super(NTXent, self).__init__()
        self.temperature = tau
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        actual_batch_size = actual_batch_size * world_size  # gather embeddings from all processes, increase batch size
        self.mask = self.mask_correlated_samples(actual_batch_size)
        self.batch_size = actual_batch_size
        self.gather_grads = gather_grads

    def mask_correlated_samples(self, batch_size):
        mask = torch.ones((batch_size * 2, batch_size * 2), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, xi, xj):

        if self.gather_grads:
            xi = torch.cat(GatherLayer.apply(xi), dim=0)
            xj = torch.cat(GatherLayer.apply(xj), dim=0)

        x = torch.cat((xi, xj), dim=0)

        x = x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-10)

        sim = self.similarity_f(x.unsqueeze(1), x.unsqueeze(0)) / self.temperature

        batch_size = int(x.size(0) / 2)

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            batch_size * 2, 1
        )

        negative_samples = sim[self.mask].reshape(batch_size * 2, -1)

        labels = torch.zeros(batch_size * 2).type_as(xi).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= 2 * batch_size

        return loss


class SIGReg(torch.nn.Module):
    """Base implementation from https://github.com/rbalestr-lab/lejepa/blob/main/MINIMAL.md"""
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 5, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        # project to 1024 based on recommendations in paper
        A = torch.randn(proj.size(-1), 1024, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t

        # permit DDP training
        x_t_cos = x_t.cos().mean(-3)
        x_t_sin = x_t.sin().mean(-3)
        avg_cos = AllReduce.apply(x_t_cos)
        avg_sin = AllReduce.apply(x_t_sin)

        err = (avg_cos - self.phi).square() + avg_sin.square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
