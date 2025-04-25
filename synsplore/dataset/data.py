from __future__ import annotations
from dataclasses import dataclass

import torch

@dataclass(repr=False)
class RouteData:
    rfeats: torch.Tensor # (N_R, F)
    pfeats: torch.Tensor # (N_P, F)
    rxnfeats: torch.Tensor # (N_RXN, F)

    ridx: torch.Tensor # (N_R, 2) Which route and which member of the sequence
    pidx: torch.Tensor # (N_P, 2) 
    rxnidx: torch.Tensor # (N_RXN, 2)
    uprodidx: torch.Tensor # (N_LP, 2)
    startidx: torch.Tensor # (N_ROUTES,)
    endidx: torch.Tensor # (N_ROUTES,)

    def __post_init__(self):
        
        if self.ridx.dim() == 1:
            self.ridx = self.ridx[:, None]
            self.ridx = torch.cat([torch.zeros(self.ridx.shape[0], 1),
                                   self.ridx], dim=1)
        
        if self.pidx.dim() == 1:
            self.pidx = self.pidx[:, None]
            self.pidx = torch.cat([torch.zeros(self.pidx.shape[0], 1),
                                   self.pidx], dim=1)
        
        if self.rxnidx.dim() == 1:
            self.rxnidx = self.rxnidx[:, None]
            self.rxnidx = torch.cat([torch.zeros(self.rxnidx.shape[0], 1),
                                     self.rxnidx], dim=1)
            
        if self.uprodidx.dim() == 1:
            self.uprodidx = self.uprodidx[:, None]
            self.uprodidx = torch.cat([torch.zeros(self.uprodidx.shape[0], 1),
                                       self.uprodidx], dim=1)

        self.rfeats = self.rfeats.to(torch.float32)
        self.pfeats = self.pfeats.to(torch.float32)
        self.rxnfeats = self.rxnfeats.to(torch.float32)
        self.pidx = self.pidx.to(torch.long)
        self.ridx = self.ridx.to(torch.long)
        self.rxnidx = self.rxnidx.to(torch.long)
        self.uprodidx = self.uprodidx.to(torch.long)
        self.startidx = self.startidx.to(torch.long)
        self.endidx = self.endidx.to(torch.long)

        assert self.rfeats.shape[0] == self.ridx.shape[0]
        assert self.pfeats.shape[0] == self.pidx.shape[0]
        assert self.rxnfeats.shape[0] == self.rxnidx.shape[0]
        assert self.startidx.shape[0] == self.endidx.shape[0]
        assert self.startidx.dim() == 1
        assert self.endidx.dim() == 1

    def clone(self):
        return RouteData(
            rfeats=self.rfeats.clone(),
            pfeats=self.pfeats.clone(),
            rxnfeats=self.rxnfeats.clone(),
            ridx=self.ridx.clone(),
            pidx=self.pidx.clone(),
            rxnidx=self.rxnidx.clone(),
            uprodidx=self.uprodidx.clone(),
            startidx=self.startidx.clone(),
            endidx=self.endidx.clone(),
        )
    
    def align_right(self):
        n_pad = self.l_max - self.endidx - 1
        self.endidx += n_pad
        self.startidx += n_pad

        prod_reps = torch.bincount(self.pidx[:, 0], 
                                    minlength=self.n_routes)
        self.pidx[:, 1] += n_pad.repeat_interleave(
            prod_reps
        )
        react_reps = torch.bincount(self.ridx[:, 0], 
                                    minlength=self.n_routes)
        self.ridx[:, 1] += n_pad.repeat_interleave(
            react_reps
        )
        rxn_reps = torch.bincount(self.rxnidx[:, 0], 
                                    minlength=self.n_routes)
        self.rxnidx[:, 1] += n_pad.repeat_interleave(
            rxn_reps
        )        
        uprod_reps = torch.bincount(self.uprodidx[:, 0], 
                                    minlength=self.n_routes)
        self.uprodidx[:, 1] += n_pad.repeat_interleave(
            uprod_reps
        )

    def encodings_to_transformer_input(self,
                                       rfeats: torch.Tensor, # (NR, D)
                                       pfeats: torch.Tensor, # (NP, D)
                                       rxnfeats: torch.Tensor, # (NRXN, D)
                                       startemb: torch.Tensor, # (D,)
                                       uprodemb: torch.Tensor, # (D,)
                                       endemb: torch.Tensor = None, # (D,)
                                       pademb: torch.Tensor = None):
        
        # Make sure all route sequences are right aligned
        self.align_right()

        # Set zeros matrix for end and pad embeddings if none
        if endemb is None:
            endemb = torch.zeros(rfeats.shape[1])
        if pademb is None:
            pademb = torch.zeros(rfeats.shape[1])

        # Check for feats to have the same number of dimensions
        assert rfeats.shape[1] == pfeats.shape[1]
        assert rfeats.shape[1] == rxnfeats.shape[1]
        assert rfeats.shape[1] == startemb.shape[0]
        assert rfeats.shape[1] == endemb.shape[0]
        assert rfeats.shape[1] == pademb.shape[0]
        d_model = rfeats.shape[1]

        # Create empty output matrix, all default to pad embedding
        out = torch.ones(self.n_routes, self.l_max, d_model) * pademb

        # Set start and end embeddings
        out[torch.arange(self.n_routes), self.startidx, :] = startemb
        out[torch.arange(self.n_routes), self.endidx, :] = endemb
        
        # Set reactant, reaction, useprod, and product features
        out[self.ridx[:, 0], self.ridx[:, 1], :] = rfeats
        out[self.pidx[:, 0], self.pidx[:, 1], :] = pfeats
        out[self.uprodidx[:, 0], self.uprodidx[:, 1], :] = uprodemb
        out[self.rxnidx[:, 0], self.rxnidx[:, 1], :] = rxnfeats

        return out

    @property
    def n_routes(self) -> int:
        return self.startidx.shape[0]
    
    @property
    def l_max(self) -> int:
        return self.endidx.max()+1
    
    def __add__(self, other: RouteData):
        rfeats = torch.cat([self.rfeats, other.rfeats], dim=0)
        pfeats = torch.cat([self.pfeats, other.pfeats], dim=0)
        rxnfeats = torch.cat([self.rxnfeats, other.rxnfeats], dim=0)

        other_ridx = other.ridx.clone()
        other_ridx[:, 0] += self.n_routes
        ridx = torch.cat([self.ridx, other_ridx], dim=0)

        other_pidx = other.pidx.clone()
        other_pidx[:, 0] += self.n_routes
        pidx = torch.cat([self.pidx, other_pidx], dim=0)

        other_rxnidx = other.rxnidx.clone()
        other_rxnidx[:, 0] += self.n_routes
        rxnidx = torch.cat([self.rxnidx, other_rxnidx], dim=0)

        other_uprodidx = other.uprodidx.clone()
        other_uprodidx[:, 0] += self.n_routes
        uprodidx = torch.cat([self.uprodidx, other_uprodidx], dim=0)

        startidx = torch.cat([self.startidx, other.startidx], dim=0)
        endidx = torch.cat([self.endidx, other.endidx], dim=0)

        return RouteData(
            rfeats=rfeats,
            pfeats=pfeats,
            rxnfeats=rxnfeats,
            pidx=pidx,
            ridx=ridx,
            rxnidx=rxnidx,
            uprodidx=uprodidx,
            startidx=startidx,
            endidx=endidx
        )
    
    def __radd__(self, other: int | RouteData):
        if other == 0:
            return RouteData(
                rfeats=self.rfeats,
                pfeats=self.pfeats,
                rxnfeats=self.rxnfeats,
                pidx=self.pidx,
                ridx=self.ridx,
                rxnidx=self.rxnidx,
                uprodidx=self.uprodidx,
                startidx=self.startidx,
                endidx=self.endidx
            )
        else:
            return self.__add__(other)
    
    def __repr__(self):
        return (
            f"RouteData["
            f"rfeats={tuple(self.rfeats.shape)}, "
            f"pfeats={tuple(self.pfeats.shape)}, "
            f"rxnfeats={tuple(self.rxnfeats.shape)}, \n"
            f"          "
            f"ridx={tuple(self.ridx.shape)}, "
            f"pidx={tuple(self.pidx.shape)}, "
            f"rxnidx={tuple(self.rxnidx.shape)}, "
            f"uprodidx={tuple(self.uprodidx.shape)}"
            f"]"
        )
    
@dataclass(repr=False)
class PharmData:
    typeids: torch.Tensor # (N_TOTAL, )
    values: torch.Tensor # (N_TOTAL, )
    idx: torch.Tensor # (N_TOTAL, ) # Which route it belongs to
    
    def __post_init__(self):

        self.typeids = self.typeids.to(torch.long)
        self.values = self.values.to(torch.float32)
        self.idx = self.idx.to(torch.long)

        assert self.typeids.shape[0] == self.values.shape[0]
        assert self.typeids.shape[0] == self.idx.shape[0]

        assert self.typeids.dim() == 1
        assert self.values.dim() == 1
        assert self.idx.dim() == 1

    def clone(self):
        return PharmData(
            typeids=self.typeids.clone(),
            values=self.values.clone(),
            idx=self.idx.clone()
        )
    
    def encodings_to_transformer_input(self,
                                       feats: torch.Tensor,
                                       pademb: torch.Tensor):

        assert feats.shape[1] == pademb.shape[0]
        d_model = feats.shape[1]
        n_pharms = self.n_pharms
        l_max = self.l_max

        # How many features each molecule has and its needed padding count
        counts = torch.bincount(self.idx, minlength=n_pharms)
        n_pad = l_max - counts

        # Create empty output with pad embeddings as default
        out = pademb.unsqueeze(0).unsqueeze(0).expand(n_pharms, 
                                                      l_max, 
                                                      d_model).clone()

        # Fill in the features
        for i in range(n_pharms):
            # Get all indices of features belonging to molecule i.
            indices = (self.idx == i).nonzero(as_tuple=True)[0]
            n_features = indices.shape[0]
            # Place the embeddings for these features, in order, at the end.
            out[i, l_max - n_features:, :] = feats[indices]

        return out


    @property
    def n_pharms(self) -> int:
        try:
            return self.idx.max()+1
        except:
            print(self.idx)
    
    @property
    def l_max(self) -> int:
        return self.idx.unique(return_counts=True)[1].max()

    def __add__(self, other: PharmData):
        typeids = torch.cat([self.typeids, other.typeids], dim=0)
        values = torch.cat([self.values, other.values], dim=0)
        other_idx = other.idx.clone()
        other_idx += self.n_pharms
        idx = torch.cat([self.idx, other_idx], dim=0)
        return PharmData(typeids=typeids, values=values, idx=idx)
    
    def __radd__(self, other: int | PharmData):
        if other == 0:
            return PharmData(
                typeids=self.typeids,
                values=self.values,
                idx=self.idx
            )
        else:
            return self.__add__(other)
    
    def __repr__(self):
        return (
            f"PharmData["
            f"typeids={tuple(self.typeids.shape)}, "
            f"values={tuple(self.values.shape)}, "
            f"idx={tuple(self.idx.shape)}"
            f"]"
        )
    
class SynsploreData:
    def __init__(self, route_data: RouteData, pharm_data: PharmData):
        self.route_data = route_data
        self.pharm_data = pharm_data

    def clone(self):
        return SynsploreData(
            route_data=self.route_data.clone(),
            pharm_data=self.pharm_data.clone()
        )
    
    def __add__(self, other: SynsploreData):
        return SynsploreData(
            route_data=self.route_data + other.route_data,
            pharm_data=self.pharm_data + other.pharm_data
        )
    
    def __radd__(self, other: int | SynsploreData):
        if other == 0:
            return SynsploreData(
                route_data=self.route_data,
                pharm_data=self.pharm_data
            )
        else:
            return self.__add__(other)
        
    def __repr__(self):
        return (
            f"SynsploreData[\n"
            f"  route_data={self.route_data}, \n"
            f"  pharm_data={self.pharm_data}\n"
            f"]"
        )
