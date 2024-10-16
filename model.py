import torch.nn as nn
from torch.distributions import Categorical
from transformers import TransfoXLModel, TransfoXLConfig, BertModel
import torch
import numpy as np
import clip
import os
# from clip.simple_tokenizer import SimpleTokenizer
from math import sqrt
import torch.nn.functional as F


class DiscreteActor(nn.Module):
    def __init__(self, input_dim, hidden, out_dim, n_hidden=0):
        super(DiscreteActor, self).__init__()
        self.modlist = [nn.Linear(input_dim, hidden),
                        nn.LayerNorm(hidden, elementwise_affine=False),
                        nn.ReLU()]
        if n_hidden > 0:
            self.modlist.extend([nn.Linear(hidden, hidden),
                                 nn.LayerNorm(hidden, elementwise_affine=False),
                                 nn.ReLU()] * n_hidden)
        self.modlist.extend([nn.Linear(hidden, out_dim),
                            nn.Softmax(dim=-1)])
        self.actor = nn.Sequential(*self.modlist).apply(orthogonal_init)

    def forward(self, states, deterministic=False):
        probs = self.actor(states)
        dist = Categorical(probs)

        if deterministic:
            action = torch.argmax(probs).squeeze()
        else:
            action = dist.sample().squeeze()

        log_prob = dist.log_prob(action)
        return action, log_prob

    def evaluate(self, states, actions):
        probs = self.actor(states)
        dist = Categorical(probs)
        log_prob = dist.log_prob(actions.squeeze())
        entropy = dist.entropy()
        return log_prob, entropy


class SmallImpalaCNN(nn.Module):
    def __init__(self, observation_shape, channel_scale=1, hidden_dim=256):
        super(SmallImpalaCNN, self).__init__()
        self.obs_size = observation_shape
        self.in_channels = 3
        kernel1 = 8 if self.obs_size[1] > 9 else 4
        kernel2 = 4 if self.obs_size[2] > 9 else 2
        stride1 = 4 if self.obs_size[1] > 9 else 2
        stride2 = 2 if self.obs_size[2] > 9 else 1
        self.block1 = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=16*channel_scale, kernel_size=kernel1, stride=stride1),
                                    nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv2d(in_channels=16*channel_scale, out_channels=32*channel_scale, kernel_size=kernel2, stride=stride2),
                                    nn.ReLU())

        in_features = self._get_feature_size(self.obs_size)
        self.fc = nn.Linear(in_features=in_features, out_features=hidden_dim)

        self.hidden_dim = hidden_dim
        self.apply(xavier_uniform_init)

    def forward(self, x):
        # print(x.shape)
        if x.shape[1] != self.in_channels:
            x = x.permute(0, 3, 1, 2)
        x = self.block1(x)
        x = self.block2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = nn.ReLU()(x)
        return x

    def _get_feature_size(self, shape):
        if shape[0] != 3:
            dummy_input = torch.zeros((shape[-1], *shape[:-1])).unsqueeze(0)
        else:
            dummy_input = torch.zeros((shape[0], *shape[1:])).unsqueeze(0)
        x = self.block2(self.block1(dummy_input))
        return np.prod(x.shape[1:])
    

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, weights, device):
        super(VectorQuantizer, self).__init__()
        self.device = device
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        # self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.weights = weights.detach()
        # self.weights_map = nn.Sequential(nn.Linear(weights.size()[0], 10000), nn.ReLU(True),
        #                                   nn.Linear(10000, 5120), nn.ReLU(True),
        #                                   nn.Linear(5120, 3000), nn.ReLU(True),
        #                                   nn.Linear(3000, 1024), nn.ReLU(True),
        #                                   nn.Linear(1024, self.n_e))
        self.weights_map = nn.Linear(weights.size()[0], n_e)

    def forward(self, z):
        weights = self.weights.permute(1, 0).contiguous()
        weights = self.weights_map(weights)
        weights = weights.permute(1, 0).contiguous()

        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # z_flattened = z
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        # d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
        #     torch.sum(self.embedding.weight**2, dim=1) - 2 * \
        #     torch.matmul(z_flattened, self.embedding.weight.t())
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(weights**2, dim=1) - 2 * \
            torch.matmul(z_flattened, weights.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(self.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        # z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        z_q = torch.matmul(min_encodings, weights).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices
    

class ResidualLayer(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1,
                      stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)]*n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x
    

class Encoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()
        kernel = 3
        stride = 2
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1,
                      stride=stride-1, padding=1),
            ResidualStack(
                h_dim, h_dim, res_h_dim, n_res_layers),
        )
    def forward(self, x):
        return self.conv_stack(x)
    
class Decoder(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, n_res_layers, res_h_dim):
        super(Decoder, self).__init__()
        kernel = 3
        stride = 2

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(
                h_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2,
                               kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim//2, out_dim, kernel_size=kernel,
                               stride=stride, padding=1)
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)
    
class VQVAE(nn.Module):
    def __init__(self, input_dim, h_dim,
        res_h_dim, n_res_layers, n_embeddings, embedding_dim, output_dim, beta, weights, device, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        self.device = device
        # encode image into continuous latent space
        self.encoder = Encoder(input_dim, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta, weights, self.device)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, output_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def calculate_zq(self, x):
        z_e = self.encoder(x)
        # z_e = z_e.unsqueeze(-1).unsqueeze(-1)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)  # TODO 计算复杂度太高，可不可以分解
        return z_q

    def forward(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)
        return embedding_loss, z_q, x_hat, perplexity
    

class LLMAgent(nn.Module):
    def __init__(self, action_dim, input_dim, optimizer, learning_rate, channel_scale=1, device=torch.device('cpu')):
        super(LLMAgent, self).__init__()
        self.device = device
        # transformer xl
        self.model_type = 'xl'
        config = TransfoXLConfig()
        self.llm = TransfoXLModel.from_pretrained('/home/mjli/projects/models/transformerxl', config=config)
        n_tokens = self.llm.word_emb.n_token
        self.word_embs = self.llm.word_emb(torch.arange(n_tokens)).detach()
        self.word_embs = self.word_embs.to(self.device)
        self.word_embs = self.word_embs[:512]
        # cv hidden to llm hidden
        self.hidden_dim = 1024

        # bert
        # self.model_type = 'bert'
        # self.llm = BertModel.from_pretrained('/home/mjli/projects/models/bert-base')
        # self.word_emb = self.llm.get_input_embeddings().weight
        # # self.word_embs = self.word_emb[:1000]
        # # self.word_embs = self.word_embs.detach().to(torch.device('cuda'))
        # self.word_embs = self.word_emb.detach().to(torch.device('cuda'))
        ## cv hidden to llm hidden
        # self.hidden_dim = 768

        # cv encoder
        self.time_encoder = SmallImpalaCNN(input_dim, channel_scale=channel_scale, hidden_dim=self.hidden_dim)

        # vq-vae
        # self.encoder = Encoder(3, self.hidden_dim, 2, 32)
        # self.pre_quantization_conv = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1, stride=1)
        # self.vector_quantization = VectorQuantizer(100, self.hidden_dim, .1, self.word_embs)
        # self.decoder = Decoder(self.hidden_dim, self.hidden_dim, 3, 2, 32)
        self.vae_model = VQVAE(input_dim=3, h_dim=self.hidden_dim, res_h_dim=32, n_res_layers=2,
                n_embeddings=100, embedding_dim=self.hidden_dim, output_dim=3, beta=.1, 
                weights=self.word_embs, device=self.device)
        self.ed_opt = getattr(torch.optim, optimizer)(self.vae_model.parameters(), lr=3e-4)

        # ac
        self.actor = DiscreteActor(self.hidden_dim * 2, 128, action_dim).apply(orthogonal_init)
        self.critic = nn.Sequential(nn.Linear(self.hidden_dim * 2, 1024),
                                    nn.LayerNorm(1024, elementwise_affine=False),
                                    nn.ReLU(),
                                    nn.Linear(1024, 512),
                                    nn.LayerNorm(512, elementwise_affine=False),
                                    nn.ReLU(),
                                    nn.Linear(512, 1)).apply(orthogonal_init)
        
        try:
            self.optimizer = getattr(torch.optim, optimizer)(self.yield_trainable_params(), lr=learning_rate)
        except AttributeError:
            raise NotImplementedError(f"{optimizer} does not exist")
        
    def yield_trainable_params(self):
        for n, p in self.named_parameters():
            if 'llm' in n or 'vae_model' in n:
                continue
            else:
                yield p

    def forward(self, states):
        bs, seqlen, *_ = states.shape
        states1 = states.reshape(bs*seqlen, *states.shape[2:])
        # print(states1.shape)
        encoded = self.vae_model.calculate_zq(states1)
        encoded = F.avg_pool2d(encoded, encoded.shape[-1], 1)
        encoded = encoded.view(bs, seqlen, -1)
        states2 = states[:, -1, :]
        time_encoded = self.time_encoder(states2)
        # time_encoded = time_encoded.view(bs, seqlen, -1)

        # print(algined_embed.shape)
        if self.model_type == 'xl':
            out = self.llm(inputs_embeds=encoded, output_hidden_states=True)
            hidden = out.last_hidden_state[:, -1, :]
        elif self.model_type == 'bert':
            out = self.llm(inputs_embeds=encoded)
            hidden = out.last_hidden_state[:, -1, :]
        # print(hidden.shape, time_encoded.shape)

        hidden = torch.concat([hidden, time_encoded], dim=-1)
        # hidden = time_encoded
        # print(hidden.shape)
        action, log_prob = self.actor(hidden)
        values = self.critic(hidden).squeeze()
        return action.cpu().numpy(), values.cpu().numpy(), log_prob.cpu().numpy().squeeze()

    def evaluate_actions(self, states, actions, detach_value_grad=False):
        # get cv rep
        bs, seqlen, *_ = states.shape
        states = states.reshape(bs*seqlen, *states.shape[2:])
        embedding_loss, encoded, x_hat, perplexity = self.vae_model(states)
        encoded = F.avg_pool2d(encoded, encoded.shape[-1], 1)
        encoded = encoded.view(bs, seqlen, -1)
        time_encoded = self.time_encoder(states)
        time_encoded = time_encoded.view(bs, seqlen, -1)

        # emb_loss & rec_loss
        recon_loss = torch.mean((x_hat - states)**2)

        # get hidden state
        if self.model_type == 'xl':
            out = self.llm(inputs_embeds=encoded, output_hidden_states=True)
            hidden = out.last_hidden_state
        elif self.model_type == 'bert':
            mask = 1 - torch.triu(torch.ones((bs, seqlen, seqlen), device=states.device), diagonal=1)  # attention_mask
            out = self.llm(inputs_embeds=encoded, attention_mask=mask)
            # out = self.llm(inputs_embeds=algined_embed)
            hidden = out.last_hidden_state
        hidden = hidden.detach()
        hidden = torch.concat([hidden, time_encoded], dim=-1)
        # hidden = time_encoded
        # hidden.detach()
        log_prob, entropy = self.actor.evaluate(hidden, actions)
        if detach_value_grad:
            hidden = hidden.detach()
        value = self.critic(hidden)
        return value, log_prob, entropy, recon_loss, embedding_loss
    

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        # print(target_embedding.size(), source_embedding.size())
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))  # 这个地方为什么要dropout
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


def orthogonal_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        torch.nn.init.orthogonal_(m.weight.data, gain=gain)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)


def xavier_uniform_init(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0.)
    return module
