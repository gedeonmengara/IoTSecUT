import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional  as F

from .base import random_state, BaseSynthesizer
from .transform import DataTransformer
from .data_sampler import DataSampler

class Discriminator(nn.Module):

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [nn.Linear(dim, item), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
            dim = item

        seq += [nn.Linear(dim, 1)]
        self.seq = nn.Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


class Residual(nn.Module):

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = nn.Linear(i, o)
        self.bn = nn.BatchNorm1d(o)
        self.relu = nn.ReLU()

    def forward(self, input_):
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(nn.Module):

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(nn.Linear(dim, data_dim))
        self.seq = nn.Sequential(*seq)

    def forward(self, input_):
        data = self.seq(input_)
        return data
    
class CGANSynthesizer(BaseSynthesizer):

    def __init__(
            self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
            generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
            discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
            log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True
    ):
        
        assert batch_size % 4 == 0

        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim

        self.generator_lr = generator_lr
        self.generator_decay = generator_decay
        self.discriminator_lr = discriminator_lr
        self.discriminator_decay = discriminator_decay

        self.batch_size = batch_size
        self.discriminator_steps = discriminator_steps
        self.log_frequency = log_frequency
        self.verbose = verbose
        self.epochs = epochs
        self.pac = pac

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self.device = torch.device(device)

        self.transformer = None
        self.data_sampler = None
        self.generator = None

    def activation(self, data):
        data_t = []
        st = 0

        for column_info in self.transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = F.gumbel_softmax(data[:, st:ed], tau=0.2, hard=False, eps=1e-10, dim=-1)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')
                
        return torch.cat(data_t, dim=1)
    
    def conditional_loss(self, data, c, m):
        loss = []
        st = 0
        st_c = 0
        for column_info in self.transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = F.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]
    
    def validate_categorical_columns(self, data, categorical_columns):
        if isinstance(data, pd.DataFrame):
            invalid_columns = set(categorical_columns) - set(data.columns)

        elif isinstance(data, np.ndarray):
            invalid_columns = []
            for column in categorical_columns:
                if column < 0 or column >= data.shape[1]:
                    invalid_columns.append(column)
        
        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')
    
    @random_state
    def train(self, train_data, categorical_columns=()):

        self.validate_categorical_columns(train_data, categorical_columns)

        epochs = self.epochs

        self.transformer = DataTransformer()
        self.transformer.fit(train_data, categorical_columns)
        train_data = self.transformer.transform(train_data)

        print("Done transform")

        self.data_sampler = DataSampler(
            train_data,
            self.transformer.output_info_list,
            self.log_frequency
        )

        print("Done samp")

        data_dim = self.transformer.output_dimensions

        self.generator = Generator(
            self.embedding_dim + self.data_sampler.dim_cond_vec(),
            self.generator_dim,
            data_dim
        ).to(self.device)

        discriminator = Discriminator(
            data_dim + self.data_sampler.dim_cond_vec(),
            self.discriminator_dim,
            pac=self.pac
        ).to(self.device)

        optimizerG = torch.optim.Adam(
            self.generator.parameters(), lr=self.generator_lr, betas=(0.5, 0.9),
            weight_decay=self.generator_decay
        )

        optimizerD = torch.optim.Adam(
            discriminator.parameters(), lr=self.discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self.discriminator_decay
        )

        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1

        steps_per_epoch = max(len(train_data) // self.batch_size, 1)
        for i in range(epochs):
            for id_ in range(steps_per_epoch):

                for n in range(self.discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self.data_sampler.sample_condvec(self.batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self.data_sampler.sample_data(self.batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self.device)
                        m1 = torch.from_numpy(m1).to(self.device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self.batch_size)
                        np.random.shuffle(perm)
                        real = self.data_sampler.sample_data(
                            self.batch_size, col[perm], opt[perm])
                        c2 = c1[perm]

                    fake = self.generator(fakez)
                    fakeact = self.activation(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self.device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    # print(fake_cat.shape)
                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self.device, self.pac)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad()
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self.data_sampler.sample_condvec(self.batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self.generator(fakez)
                fakeact = self.activation(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self.conditional_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()

            if self.verbose:
                print(f'Epoch {i+1}, Loss G: {loss_g.detach().cpu(): .4f},'  # noqa: T001
                      f'Loss D: {loss_d.detach().cpu(): .4f}',
                      flush=True)
                
    @random_state
    def generate(self, n, condition_column=None, condition_value=None):

        if condition_column is not None and condition_value is not None:
            condition_info = self.transformer.convert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self.data_sampler.generate_cond_from_condition_column_info(
                condition_info, self.batch_size)
        else:
            global_condition_vec = None

        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self.device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self.data_sampler.sample_original_condvec(self.batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self.device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self.generator(fakez)
            fakeact = self.activation(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self.transformer.inverse_transform(data)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self.device = device
        if self.generator is not None:
            self.generator.to(self.device)