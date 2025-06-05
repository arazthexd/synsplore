from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

class SynModule(nn.Module):
    def __init__(self,
                 d_model: int,
                 r_enc: nn.Module, # _SimpleMLP
                 rxn_enc: nn.Module, # _SimpleEmbedding
                 pos_enc: nn.Module, # TODO
                 seq_enc: nn.Module, # TODO
                 classifier: nn.Module, # _SimpleMLP
                 r_dec: nn.Module, # _SimpleVAEDecoder
                 rxn_dec: nn.Module): # _SimpleVAEDecoder
        super().__init__()
        print(f"r_enc: {r_enc}")
        self.d_model = d_model
        self.r_enc = r_enc
        self.rxn_enc = rxn_enc
        self.usep_emb = nn.Parameter(torch.randn(d_model))
        self.st_emb = nn.Parameter(torch.randn(d_model))
        self.pos_enc = pos_enc
        self.seq_enc = seq_enc
        self.classifier = classifier
        self.r_dec = r_dec
        self.rxn_dec = rxn_dec

    def process(self, syndata: TensorDict, cond: torch.Tensor = None):
        renc, rxnenc = self._get_encs(syndata)

        syndata["renc"] = renc
        syndata["rxnenc"] = rxnenc
        seqenc = self._get_seqenc(syndata)
        
        syndata["seqenc"] = seqenc
        syndata["seqenc"] = self._update_seqenc(syndata, cond)

        cls_idx, r_idx, rxn_idx = self._get_predidx(syndata)
        return syndata, cls_idx, r_idx, rxn_idx
    
    def forward(self, syndata: TensorDict, cond: torch.Tensor = None):
        syndata, cls_idx, r_idx, rxn_idx = self.process(syndata, cond)
        
        seqenc = syndata["seqenc"]
        cls_pred = self.classifier(seqenc[*cls_idx.T])
        r_pred = self.r_dec(seqenc[*r_idx.T])
        rxn_pred = self.rxn_dec(seqenc[*rxn_idx.T])

        return cls_pred, r_pred, rxn_pred

    def get_loss(self, 
                 syndata: TensorDict | Dict[str, torch.Tensor], # for typing :)
                 cond: torch.Tensor = None):
        
        # syndata, cls_idx, r_idx, rxn_idx = self.process(syndata, cond)

        cls_true, r_true, rxn_true = self.get_trues(syndata)

        loss_cls = self.classifier.get_loss(cls_true)
        
        loss_r = self.r_dec.get_loss(r_true)

        loss_rxn = self.rxn_dec.get_loss(rxn_true)
        
        return loss_cls, loss_r, loss_rxn
    
    @staticmethod
    def get_trues(syndata: TensorDict | Dict[str, torch.Tensor]):
        
        cls_true = torch.cat([
            torch.tensor([[1, 0, 0, 0]]).repeat(syndata["ridx"].shape[0], 1),
            torch.tensor([[0, 1, 0, 0]]).repeat(syndata["rxnidx"].shape[0], 1),
            torch.tensor([[0, 0, 1, 0]]).repeat(syndata["usepidx"].shape[0], 1),
            torch.tensor([[0, 0, 0, 1]]).repeat(syndata["endidx"].shape[0], 1),
        ]).type(torch.float)

        if "rout" in syndata.keys():
            rout = syndata["rout"]
        else:
            rout = syndata["rfeats"]
        
        if "rxnout" in syndata.keys():
            rxnout = syndata["rxnout"]
        else:
            rxnout = syndata["rxnfeats"]

        return (cls_true, rout, rxnout)

    def get_metrics(self, 
                    syndata: TensorDict | Dict[str, torch.Tensor]):
        cls_true, r_true, rxn_true = self.get_trues(syndata)

        cls_metrics = self.classifier.get_metrics(cls_true)
        r_metrics = self.r_dec.get_metrics(r_true)
        rxn_metrics = self.rxn_dec.get_metrics(rxn_true)

        metrics_outputs = {"cls_metrics": cls_metrics,
                           "r_metrics": r_metrics,
                           "rxn_metrics": rxn_metrics}
        return metrics_outputs

    def _get_encs(self, syndata: TensorDict):
        renc = self.r_enc(syndata["rfeats"])
        rxnenc = self.rxn_enc(syndata["rxnfeats"])
        return renc, rxnenc
    
    def _get_seqenc(self, syndata: TensorDict):
        syndata["usepemb"] = self.usep_emb
        syndata["startemb"] = self.st_emb
        syndata["endemb"] = torch.zeros(self.d_model)
        syndata["pademb"] = torch.zeros(self.d_model)

        seqenc = self._construct_transformer_seq(syndata)
        seqenc = seqenc[:, :-1]
        seqenc = self.pos_enc(seqenc, syndata["stidx"])
        return seqenc
    
    def _update_seqenc(self, syndata: TensorDict, cond: torch.Tensor = None):
        pad_mask = self._construct_mask(syndata, 
                                        ones_by_default=True,
                                        reverse_start=True,
                                        reverse_ridx=True,
                                        reverse_rxnidx=True,
                                        reverse_usepidx=True)
        seqenc = self.seq_enc(syndata["seqenc"], pad_mask, cond)
        return seqenc

    def _get_predidx(self, syndata: TensorDict):
        cls_idx = torch.cat([
            syndata["ridx"], syndata["rxnidx"], 
            syndata["usepidx"], 
            torch.stack([torch.arange(syndata["endidx"].shape[0]),
                          syndata["endidx"]], dim=1), 
        ], dim=0)
        cls_idx[:, 1] -= 1
        
        r_idx: torch.Tensor = syndata["ridx"].clone()
        r_idx[:, 1] -= 1

        rxn_idx: torch.Tensor = syndata["rxnidx"].clone()
        rxn_idx[:, 1] -= 1

        return cls_idx, r_idx, rxn_idx


    @staticmethod
    def _construct_transformer_seq(
        syndata: TensorDict | Dict[str, torch.Tensor]):

        d_model = syndata["renc"].shape[1]
        n = syndata["stidx"].shape[0]
        l = int(max(torch.cat([syndata["ridx"][:, 1], 
                               syndata["rxnidx"][:, 1], 
                               syndata["usepidx"][:, 1], 
                               syndata["stidx"], 
                               syndata["endidx"]]))) + 1
        
        seq = torch.tile(syndata["pademb"], (n, l, 1))
        seq[*syndata["ridx"].T] = syndata["renc"]
        seq[*syndata["rxnidx"].T] = syndata["rxnenc"]
        seq[*syndata["usepidx"].T] = syndata["usepemb"]
        seq[torch.arange(n), syndata["stidx"]] = syndata["startemb"]
        seq[torch.arange(n), syndata["endidx"]] = syndata["endemb"]
        return seq
    
    @staticmethod
    def _construct_mask(data: TensorDict,
                        ones_by_default: bool = True,
                        reverse_start: bool = False,
                        reverse_end: bool = False,
                        reverse_ridx: bool = False,
                        reverse_rxnidx: bool = False,
                        reverse_usepidx: bool = False,
                        seqid_shift: int = 0):
        seq = data["seqenc"]
        
        if ones_by_default:
            mask = torch.ones(seq.shape[0], seq.shape[1])
        else:
            mask = torch.zeros(seq.shape[0], seq.shape[1])

        if ones_by_default:
            mask_vals = 0
        else:
            mask_vals = 1
        
        if reverse_ridx:
            idx = data["ridx"].clone()
            idx[:, 1] += seqid_shift
            mask[*idx.T] = mask_vals
        if reverse_rxnidx:
            idx = data["rxnidx"].clone()
            idx[:, 1] += seqid_shift
            mask[*idx.T] = mask_vals
        if reverse_usepidx:
            idx = data["usepidx"].clone()
            idx[:, 1] += seqid_shift
            mask[*idx.T] = mask_vals
        if reverse_start:
            idx = data["stidx"].clone()
            idx += seqid_shift
            mask[torch.arange(seq.shape[0]), idx] = mask_vals
        if reverse_end:
            idx = data["endidx"].clone()
            idx += seqid_shift
            mask[torch.arange(seq.shape[0]), idx] = mask_vals
        return mask
        
