

import torch
import torch.nn as nn
from tensordict import TensorDict

from .blocks import (
    SequenceEncoder, SimpleEmbedding, 
    SimpleMLP, PositionalEncoder
)

class SynsploreDecoder(nn.Module): 
    def __init__(self, 
                 r_indim: int, rxn_indim: int, 
                 d_model: int, 
                 r_outdim: int, rxn_outdim: int,
                 has_encoder_input: bool = False): 
        super().__init__()
        self.r_indim = r_indim
        self.rxn_indim = rxn_indim,
        self.d_model = d_model
        self.r_outdim = r_outdim
        self.rxn_outdim = rxn_outdim
        self.has_encoder_input = has_encoder_input

        self.usepemb = nn.Parameter(torch.randn(d_model))
        self.startemb = nn.Parameter(torch.randn(d_model))

        self.r_enc = SimpleMLP(r_indim, d_model)
        self.rxn_enc = SimpleEmbedding(rxn_indim, d_model)
        self.pos_enc = PositionalEncoder(d_model)
        self.seq_enc = SequenceEncoder(d_model)
        self.classifier = nn.Sequential(
            SimpleMLP(d_model, 4, n_layers=1),
            nn.Softmax(dim=-1)
        )
        self.r_dec = SimpleMLP(d_model, r_outdim, n_layers=1) # TODO: VAE-Decoder
        self.rxn_dec = SimpleMLP(d_model, rxn_outdim, n_layers=1) # TODO: above
        
    def forward(self, data: TensorDict):
        
        seq = self.prepare_seq(data)
        
        tf_inp = seq[:, :-1]
        pad_mask = self._construct_padding_mask(tf_inp, data)
        out = self.seq_enc.forward(tf_inp, pad_mask)

        cls_pred = self.classifier.forward(out)

        out_ridx = data["ridx"].clone()
        out_ridx[:, 1] -= 1
        ry_pred = self.r_dec.forward(out[*out_ridx.T])

        out_rxnidx = data["rxnidx"].clone()
        out_rxnidx[:, 1] -= 1
        rxny_pred = self.rxn_dec.forward(out[*out_rxnidx.T])

        # is_r = torch.zeros((inp.shape[0], inp.shape[1]), dtype=torch.bool)
        # is_r[*ridx.T] = 1
        # is_rxn = torch.zeros((inp.shape[0], inp.shape[1]), dtype=torch.bool)
        # is_rxn[*rxnidx.T] = 1
        # is_usep = torch.zeros((inp.shape[0], inp.shape[1]), dtype=torch.bool)
        # is_usep[*usepidx.T] = 1
        # classifier_ytrue = torch.stack([is_r, is_rxn, is_usep], dim=-1)

        return cls_pred, ry_pred, rxny_pred
    
    def infere_step(self, data: TensorDict):
        seq = self.prepare_seq(data)
        seq = seq[:, :-1]
        pad_mask = self._construct_padding_mask(seq, data)
        out = self.seq_enc.forward(seq, pad_mask=pad_mask)

        last = out[:, -1]
        cls_pred: torch.Tensor = self.classifier.forward(last)
        cls_pred = cls_pred.argmax(dim=-1)
        
        ridx: torch.Tensor = torch.where(cls_pred == 0)[0]
        rfeats: torch.Tensor = self.r_dec.forward(last[ridx])

        rxnidx: torch.Tensor = torch.where(cls_pred == 1)[0]
        rxnfeats: torch.Tensor = self.rxn_dec.forward(last[rxnidx])

        usepidx: torch.Tensor = torch.where(cls_pred == 2)[0]
        endidx: torch.Tensor = torch.where(cls_pred == 3)[0]
        
        return (ridx, rfeats), (rxnidx, rxnfeats), usepidx, endidx

    def prepare_seq(self, data: TensorDict):
        
        rfeats: torch.Tensor = data["rfeats"]
        rxnfeats: torch.Tensor = data["rxnfeats"]
        ridx: torch.Tensor = data["ridx"]
        rxnidx: torch.Tensor = data["rxnidx"]
        usepidx: torch.Tensor = data["usepidx"]
        stidx: torch.Tensor = data["stidx"]
        endidx: torch.Tensor = data["endidx"]
        
        renc = self.r_enc.forward(rfeats)
        rxnenc = self.rxn_enc.forward(rxnfeats)
        seq = self._construct_transformer_seq(
            renc=renc,
            rxnenc=rxnenc,
            usepemb=self.usepemb,
            startemb=self.startemb,
            endemb=torch.zeros(self.d_model),
            pademb=torch.zeros(self.d_model),
            ridx=ridx,
            rxnidx=rxnidx,
            usepidx=usepidx,
            startidx=stidx,
            endidx=endidx
        )
        seq = self.pos_enc.forward(seq)
        return seq
    
    def _construct_padding_mask(self, seq: torch.Tensor, data: TensorDict):
        pad_mask = torch.ones(seq.shape[0], seq.shape[1])
        pad_mask[*data["ridx"].T] = 0
        pad_mask[*data["rxnidx"].T] = 0
        pad_mask[*data["usepidx"].T] = 0
        pad_mask[torch.arange(seq.shape[0]), data["stidx"]] = 0
        return pad_mask
    
    @staticmethod
    def _construct_transformer_seq(renc: torch.Tensor,
                                   rxnenc: torch.Tensor,
                                   usepemb: torch.Tensor,
                                   startemb: torch.Tensor,
                                   endemb: torch.Tensor,
                                   pademb: torch.Tensor,
                                   ridx: torch.Tensor,
                                   rxnidx: torch.Tensor,
                                   usepidx: torch.Tensor,
                                   startidx: torch.Tensor,
                                   endidx: torch.Tensor):
        d_model = renc.shape[1]
        n = startidx.shape[0]
        l = int(max(torch.cat([ridx[:, 1], 
                               rxnidx[:, 1], 
                               usepidx[:, 1], 
                               startidx, 
                               endidx]))) + 1
        
        seq = torch.tile(pademb, (n, l, 1))
        seq[*ridx.T] = renc
        seq[*rxnidx.T] = rxnenc
        seq[*usepidx.T] = usepemb
        seq[torch.arange(n), startidx] = startemb
        seq[torch.arange(n), endidx] = endemb
        return seq

class SynsploreED(nn.Module):
    pass

if __name__ == "__main__":
    print(SimpleMLP(10, 20)(torch.randn((100, 10))).shape)
    print(SimpleEmbedding(10, 20)(torch.randint(0, 9, (100,))).shape)
    print(SimpleEmbedding(10, 20)(torch.randn((100, 10))).shape)
    
    

        
        