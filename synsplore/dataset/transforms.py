from typing import Dict

import torch
from tensordict import TensorDict

class BaseTransform:
    def __call__(self, td: TensorDict) -> TensorDict:
        return self.transform(td)
    
    def transform(self, td: TensorDict) -> TensorDict:
        pass

class NumericsToDType(BaseTransform):
    def __init__(self, dtype=torch.dtype):
        super().__init__()
        self.dtype = dtype
    
    def transform(self, td):
        syndata, pharmdata = td["syndata"], td["pharmdata"]
        
        syndata["rfeats"] = syndata["rfeats"].type(self.dtype)
        syndata["rxnfeats"] = syndata["rxnfeats"].type(self.dtype)
        syndata["pfeats"] = syndata["pfeats"].type(self.dtype)

        pharmdata["dists"] = pharmdata["dists"].type(self.dtype)
        pharmdata["angles"] = pharmdata["angles"].type(self.dtype)
        
        return td
    
class IndicesToDtype(BaseTransform):
    pass # TODO

class SynRightAlign(BaseTransform):
    def transform(self, td: TensorDict):
        syndata: Dict[str, torch.Tensor] = td["syndata"]
        
        n = syndata["stidx"].shape[0]
        l_max = int(max(torch.cat([syndata["ridx"][:, 1], 
                                   syndata["rxnidx"][:, 1], 
                                   syndata["pidx"][:, 1],
                                   syndata["usepidx"][:, 1], 
                                   syndata["stidx"], 
                                   syndata["endidx"]]))) + 1
        n_pad = l_max - syndata["endidx"] - 1

        p_reps = torch.bincount(syndata["pidx"][:, 0], 
                                minlength=n)
        syndata["pidx"][:, 1] += n_pad.repeat_interleave(
            p_reps
        )
        
        r_reps = torch.bincount(syndata["ridx"][:, 0], 
                                minlength=n)
        syndata["ridx"][:, 1] += n_pad.repeat_interleave(
            r_reps
        )
        
        rxn_reps = torch.bincount(syndata["rxnidx"][:, 0], 
                                  minlength=n)
        syndata["rxnidx"][:, 1] += n_pad.repeat_interleave(
            rxn_reps
        )

        usep_reps = torch.bincount(syndata["usepidx"][:, 0], 
                                   minlength=n)
        syndata["usepidx"][:, 1] += n_pad.repeat_interleave(
            usep_reps
        )

        syndata["stidx"] += n_pad
        syndata["endidx"] += n_pad
    
        return td
