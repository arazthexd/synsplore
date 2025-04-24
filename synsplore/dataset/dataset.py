from __future__ import annotations
from typing import List, Tuple
import dill
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from druglab.synthesis import SynthesisRoute, SynRouteStorage, ActionTypes
from druglab.pharm import Pharmacophore

from .data import RouteData, PharmData, SynsploreData

class SynsploreDataset(Dataset):
    def __init__(self, 
                 routes_path: str,
                 pharms_path: str):
        
        with open(routes_path, 'rb') as f:
            self.routes: SynRouteStorage = dill.load(f)

        with open(pharms_path, 'rb') as f:
            self.pharms: List[List[Pharmacophore]] = dill.load(f)

        self.mappings: List[Tuple[int, int]] = [
            (i, j) 
            for i in range(len(self.routes)) 
            for j in range(len(self.pharms[i]))
        ]

    def __len__(self):
        return len(self.mappings)
    
    def __getitem__(self, idx):
        route_idx, pharm_idx = self.mappings[idx]
        route: SynthesisRoute = self.routes[route_idx]
        pharm: Pharmacophore = self.pharms[route_idx][pharm_idx]

        # Route...
        ridx = [i for i, member in enumerate(route.seq)
                if member.type == ActionTypes.REACTANT]
        rfeats = torch.tensor(self.routes.rstore.feats[ridx])
        ridx = torch.tensor(ridx)

        pidx = [i for i, member in enumerate(route.seq)
                if member.type == ActionTypes.PRODUCT]
        pfeats = torch.tensor(self.routes.pstore.feats[pidx])
        pidx = torch.tensor(pidx)

        rxnidx = [i for i, member in enumerate(route.seq)
                  if member.type == ActionTypes.REACTION]
        rxnfeats = torch.tensor(self.routes.rxnstore.feats[rxnidx])
        rxnidx = torch.tensor(rxnidx)

        startidx = torch.tensor([0])
        endidx = torch.tensor([len(route.seq) - 1])

        route_data = RouteData(
            rfeats=rfeats,
            pfeats=pfeats,
            rxnfeats=rxnfeats,
            pidx=pidx,
            ridx=ridx,
            rxnidx=rxnidx,
            startidx=startidx,
            endidx=endidx,
        )

        # Pharmacophore...
        typeids = torch.tensor(pharm.distances.tyidx)
        values = torch.tensor(pharm.distances.dist)
        idx = torch.arange(typeids.shape[0])

        pharm_data = PharmData(
            typeids=typeids,
            values=values,
            idx=idx
        )

        return SynsploreData(route_data=route_data, 
                             pharm_data=pharm_data)