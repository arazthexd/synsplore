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

        self.rfeats = self.rfeats.to(torch.float32)
        self.pfeats = self.pfeats.to(torch.float32)
        self.rxnfeats = self.rxnfeats.to(torch.float32)
        self.pidx = self.pidx.to(torch.long)
        self.ridx = self.ridx.to(torch.long)
        self.rxnidx = self.rxnidx.to(torch.long)

        assert self.rfeats.shape[0] == self.ridx.shape[0]
        assert self.pfeats.shape[0] == self.pidx.shape[0]
        assert self.rxnfeats.shape[0] == self.rxnidx.shape[0]
        assert self.startidx.shape[0] == self.endidx.shape[0]
        assert self.startidx.dim() == 1
        assert self.endidx.dim() == 1

    @property
    def n_routes(self) -> int:
        return self.startidx.shape[0]
    
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

        startidx = torch.cat([self.startidx, other.startidx], dim=0)
        endidx = torch.cat([self.endidx, other.endidx], dim=0)

        return RouteData(
            rfeats=rfeats,
            pfeats=pfeats,
            rxnfeats=rxnfeats,
            pidx=pidx,
            ridx=ridx,
            rxnidx=rxnidx,
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
            f"rxnfeats={tuple(self.rxnfeats.shape)}, "
            f"pidx={tuple(self.pidx.shape)}, "
            f"ridx={tuple(self.ridx.shape)}, "
            f"rxnidx={tuple(self.rxnidx.shape)}"
            f"]"
        )
    
@dataclass(repr=False)
class PharmData:
    typeids: torch.Tensor # (N_TOTAL, )
    values: torch.Tensor # (N_TOTAL, )
    idx: torch.Tensor # (N_TOTAL, 2) # Which pharmacophore, seq idx

    def __post_init__(self):
        if self.idx.dim() == 1:
            self.idx = self.idx[:, None]
            self.idx = torch.cat([torch.ones(self.idx.shape[0], 1),
                                  self.idx], dim=1)
            
        self.typeids = self.typeids.to(torch.long)
        self.values = self.values.to(torch.float32)
        self.idx = self.idx.to(torch.long)

        assert self.typeids.shape[0] == self.values.shape[0]
        assert self.typeids.shape[0] == self.idx.shape[0]
        assert self.idx.shape[1] == 2

        assert self.typeids.dim() == 1
        assert self.values.dim() == 1
        assert self.idx.dim() == 2

    def __add__(self, other: PharmData):
        typeids = torch.cat([self.typeids, other.typeids], dim=0)
        values = torch.cat([self.values, other.values], dim=0)
        other_idx = other.idx.clone()
        other_idx[:, 0] += self.idx[:, 0].max() + 1
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
