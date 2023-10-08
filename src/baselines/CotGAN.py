import torch
import torch


from src.baselines.base import BaseTrainer
from tqdm import tqdm
from torch.nn.functional import one_hot
import wandb
import torch.optim.swa_utils as swa_utils


class COTGANTrainer(BaseTrainer):
    def __init__(self, D_h, D_m, G, sinkhorn_eps, sinkhorn_l, train_dl, config,
                 **kwargs):
        super(COTGANTrainer, self).__init__(
            G=G,
            G_optimizer=torch.optim.Adam(
                G.parameters(), lr=config.lr_G, betas=(0, 0.9)),
            **kwargs
        )

        self.config = config
        self.D_steps_per_G_step = config.D_steps_per_G_step
        self.D_h = D_h
        self.D_m = D_m
        self.D_h_optimizer = torch.optim.Adam(
            self.D_h.parameters(), lr=config.lr_D, betas=(0, 0.9))
        self.D_m_optimizer = torch.optim.Adam(
            self.D_m.parameters(), lr=config.lr_D, betas=(0, 0.9))  # Using TTUR
        self.sinkhorn_eps = sinkhorn_eps
        self.sinkhor_l = sinkhorn_l

        self.train_dl = train_dl
        self.conditional = self.config.conditional
        self.reg_param = 0
        self.losses_history
        self.averaged_G = swa_utils.AveragedModel(G)

    def fit(self, device):
        self.G.to(device)
        self.D_m.to(device)
        self.D_h.to(device)
        #wandb.watch(models=self.D, log='all', log_freq=10, log_graph=False)

        for i in tqdm(range(self.n_gradient_steps)):
            self.step(device, i)
            if i > self.config.swa_step_start:
                self.averaged_G.update_parameters(self.G)

    def step(self, device, step):
        for i in range(self.D_steps_per_G_step):
            # generate x_fake

            if self.conditional:
                data = next(iter(self.train_dl))
                x = data[0].to(device)
                condition = one_hot(
                    data[1], self.config.num_classes).unsqueeze(1).repeat(1, data[0].shape[1], 1).to(device)
                x_real_batch = torch.cat(
                    [x, condition], dim=2)
            else:
                condition = None
                x_real1 = next(iter(self.train_dl))[0].to(device)
                x_real2 = next(iter(self.train_dl))[0].to(device)
            with torch.no_grad():
                x_fake = self.G(batch_size=self.batch_size,
                                n_lags=self.config.n_lags, condition=condition, device=device)
                x_fake_p = self.G(batch_size=self.batch_size,
                                  n_lags=self.config.n_lags, condition=condition, device=device)

            D_loss = self.D_trainstep(
                x_fake, x_fake_p, x_real1, x_real2)
            if i == 0:
                self.losses_history['D_loss'].append(D_loss)
                wandb.log({'D_loss': D_loss}, step)
        G_loss = self.G_trainstep(x_real1, x_real2, device, step)
        wandb.log({'G_loss': G_loss}, step)

    def G_trainstep(self, x_real, x_real_p, device, step):
        if self.conditional:
            condition = one_hot(torch.randint(
                0, self.config.num_classes, (self.batch_size,)),
                self.config.num_classes).float().unsqueeze(1).repeat(1, self.config.n_lags, 1).to(device)

        else:
            condition = None
        x_fake = self.G(batch_size=self.batch_size,
                        n_lags=self.config.n_lags, condition=condition, device=device)
        x_fake_p = self.G(batch_size=self.batch_size,
                          n_lags=self.config.n_lags, condition=condition, device=device)

        toggle_grad(self.G, True)
        self.G.train()
        self.G_optimizer.zero_grad()
        h_real_p, h_fake, h_fake_p = self.D_h(
            x_real_p), self.D_h(x_fake), self.D_h(x_fake_p)
        m_real, m_real_p, m_fake = self.D_m(
            x_real), self.D_m(x_real_p), self.D_m(x_fake)

        self.D_m.train()
        self.D_h.train()
        G_loss = compute_mixed_sinkhorn_loss(x_real, x_fake, m_real, m_fake,
                                             h_fake, self.sinkhorn_eps, self.sinkhor_l,
                                             x_real_p, x_fake_p, m_real_p,
                                             h_real_p, h_fake_p)
        G_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.G.parameters(), self.config.grad_clip)
        self.losses_history['G_loss'].append(G_loss)
        self.G_optimizer.step()
        self.evaluate(x_fake, x_real, step, self.config)

        return G_loss.item()

    def D_trainstep(self, x_fake, x_fake_p, x_real, x_real_p):
        toggle_grad(self.D_m, True)
        toggle_grad(self.D_h, True)
        self.D_m.train()
        self.D_h.train()
        self.D_m_optimizer.zero_grad()
        self.D_h_optimizer.zero_grad()

        # On real data

        # On fake data
        x_fake.requires_grad_()
        h_real_p, h_fake, h_fake_p = self.D_h(
            x_real_p), self.D_h(x_fake), self.D_h(x_fake_p)
        m_real, m_real_p, m_fake = self.D_m(
            x_real), self.D_m(x_real_p), self.D_m(x_fake)

        dloss = compute_mixed_sinkhorn_loss(x_real, x_fake, m_real, m_fake,
                                            h_fake, self.sinkhorn_eps, self.sinkhor_l,
                                            x_real_p, x_fake_p, m_real_p,
                                            h_real_p, h_fake_p)

        # Compute regularizer on fake / real

        dloss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(
            self.D_m.parameters(), self.config.grad_clip)
        # Step discriminator params
        self.D_m_optimizer.step()
        torch.nn.utils.clip_grad_norm_(
            self.D_h.parameters(), self.config.grad_clip)
       # dloss.backward(retain_graph=True)
        # Step discriminator params
        self.D_h_optimizer.step()

        # Toggle gradient to False
        toggle_grad(self.D_m, False)
        toggle_grad(self.D_h, False)

        return dloss.item()


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def cost_matrix(x, y, p=2):
    '''
    L2 distance between vectors, using expanding and hence is more memory intensive
    :param x: x is tensor of shape [batch_size, time steps, features]
    :param y: y is tensor of shape [batch_size, time steps, features]
    :param p: power
    :return: cost matrix: a matrix of size [batch_size, batch_size] where
    '''
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    b = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
    c = torch.sum(b, -1)
    return c


def modified_cost(x, y, h, M):
    '''
    :param x: a tensor of shape [batch_size, time steps, features]
    :param y: a tensor of shape [batch_size, time steps, features]
    :param h: a tensor of shape [batch size, time steps, J]
    :param M: a tensor of shape [batch size, time steps, J]
    :param scaling_coef: a scaling coefficient for squared distance between x and y
    :return: L1 cost matrix plus h, M modification:
    a matrix of size [batch_size, batch_size] where
    c_hM_{ij} = c_hM(x^i, y^j) = L2_cost + \sum_{t=1}^{T-1}h_t\Delta_{t+1}M
    ====> NOTE: T-1 here, T = # of time steps
    '''
    # compute sum_{t=1}^{T-1} h[t]*(M[t+1]-M[t])
    DeltaMt = M[:, 1:, :] - M[:, :-1, :]
    ht = h[:, :-1, :]
    time_steps = ht.shape[1]
    sum_over_j = torch.sum(ht[:, None, :, :] * DeltaMt[None, :, :, :], -1)
    C_hM = torch.sum(sum_over_j, -1) / time_steps

    # Compute L2 cost $\sum_t^T |x^i_t - y^j_t|^2$
    cost_xy = cost_matrix(x, y)

    return cost_xy + C_hM


def compute_sinkhorn(x, y, h, M, epsilon=0.1, niter=10):
    """
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    """
    n = x.shape[0]

    # The Sinkhorn algorithm takes as input three variables :
    C = modified_cost(x, y, h, M)  # shape: [batch_size, batch_size]b

    # both marginals are fixed with equal weights
    # mu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
    # nu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
    mu = 1. / n * torch.ones(n, requires_grad=False, device=x.device)
    nu = 1. / n * torch.ones(n, requires_grad=False, device=x.device)

    # Parameters of the Sinkhorn algorithm.
    rho = 1  # (.5) **2          # unbalanced transport
    tau = -.8  # nesterov-like acceleration
    lam = rho / (rho + epsilon)  # Update exponent
    thresh = 10**(-4)  # stopping criterion

    # Elementary operations .....................................................................
    def ave(u, u1):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    def lse(A):
        "log-sum-exp"
        # add 10^-6 to prevent NaN
        return torch.logsumexp(A, dim=-1, keepdim=True)

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.
    # to check if algorithm terminates because of threshold or max iterations reached
    actual_nits = 0

    for i in range(niter):
        u1 = u  # useful to check the update
        u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
        # accelerated unbalanced iterations
        # u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze()   ) + u ) )
        # v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze() ) + v ) )
        err = (u - u1).abs().sum()

        actual_nits += 1
        if (err < thresh).item():
            break
    U, V = u, v
    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    cost = torch.sum(pi * C)  # Sinkhorn cost

    return cost


def scale_invariante_martingale_regularization(M, reg_lam):
    '''
    Compute the regularization for the martingale condition (i.e. p_M).
    :param M: a tensor of shape (batch_size, sequence length), the output of an RNN applied to X
    :param reg_lam: scale parameter for first term in pM
    :return: A rank 0 tensors (i.e. scalers)
    This tensor represents the martingale penalization term denoted $p_M$
    '''
    m, t, j = M.shape
    # m = torch.tensor(m).type(torch.FloatTensor)
    # t = torch.tensor(m).type(torch.FloatTensor)
    # compute delta M matrix N
    N = M[:, 1:, :] - M[:, :-1, :]
    N_std = N / (torch.std(M, (0, 1)) + 1e-06)

    # Compute \sum_i^m(\delta M)
    sum_m_std = torch.sum(N_std, 0) / m
    # Compute martingale penalty: P_M1 =  \sum_i^T(|\sum_i^m(\delta M)|) * scaling_coef
    sum_across_paths = torch.sum(torch.abs(sum_m_std)) / t
    # the total pM term
    pm = reg_lam * sum_across_paths
    return pm


def compute_mixed_sinkhorn_loss(f_real, f_fake, m_real, m_fake, h_fake, sinkhorn_eps, sinkhorn_l,
                                f_real_p, f_fake_p, m_real_p, h_real_p, h_fake_p, scale=False):
    '''
    :param x and x'(f_real, f_real_p): real data of shape [batch size, time steps, features]
    :param y and y'(f_fake, f_fake_p): fake data of shape [batch size, time steps, features]
    :param h and h'(h_real, h_fake): h(y) of shape [batch size, time steps, J]
    :param m and m'(m_real and m_fake): M(x) of shape [batch size, time steps, J]
    :param scaling_coef: a scaling coefficient
    :param sinkhorn_eps: Sinkhorn parameter - epsilon
    :param sinkhorn_l: Sinkhorn parameter - the number of iterations
    :return: final Sinkhorn loss(and actual number of sinkhorn iterations for monitoring the training process)
    '''
    f_real = f_real.reshape(f_real.shape[0], f_real.shape[1], -1)
    f_fake = f_fake.reshape(f_fake.shape[0], f_fake.shape[1], -1)
    f_real_p = f_real_p.reshape(f_real_p.shape[0], f_real_p.shape[1], -1)
    f_fake_p = f_fake_p.reshape(f_fake_p.shape[0], f_fake_p.shape[1], -1)
    loss_xy = compute_sinkhorn(
        f_real, f_fake, h_fake, m_real, sinkhorn_eps, sinkhorn_l)
    loss_xyp = compute_sinkhorn(
        f_real_p, f_fake_p, h_fake_p, m_real_p, sinkhorn_eps, sinkhorn_l)
    loss_xx = compute_sinkhorn(
        f_real, f_real_p, h_real_p, m_real, sinkhorn_eps, sinkhorn_l)
    loss_yy = compute_sinkhorn(
        f_fake, f_fake_p, h_fake_p, m_fake, sinkhorn_eps, sinkhorn_l)

    loss = loss_xy + loss_xyp - loss_xx - loss_yy
    return loss
