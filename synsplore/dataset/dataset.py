from __future__ import annotations
from typing import List, Tuple, Set
import dill
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from druglab.synthesis import (
    SynthesisRoute, SynRouteStorage, 
    ActionTypes, _SequenceMember
)
from druglab.pharm import Pharmacophore

from .data import RouteData, PharmData, SynsploreData

class SynsploreDataset(Dataset):
    def __init__(self, 
                 routes_path: str,
                 pharms_path: str,
                 enabled: List[int] | Set[int] = None):
        
        with open(routes_path, 'rb') as f:
            self.routes: SynRouteStorage = dill.load(f)

        with open(pharms_path, 'rb') as f:
            self.pharms: List[List[Pharmacophore]] = dill.load(f)

        self.mappings: List[Tuple[int, int]] = [
            (i, j) 
            for i in range(len(self.routes)) 
            for j in range(len(self.pharms[i]))
        ]

        if enabled is None:
            enabled: Set[int] = [
                ActionTypes.START,
                ActionTypes.REACTANT,
                ActionTypes.REACTION,
                ActionTypes.PRODUCT,
                ActionTypes.USEPROD,
                ActionTypes.END
            ]
        self.enabled = set(enabled)

    def __len__(self):
        return len(self.mappings)
    
    def __getitem__(self, idx):
        route_idx, pharm_idx = self.mappings[idx]
        route_seq: List[_SequenceMember] = self.routes.seqs[route_idx]
        pharm: Pharmacophore = self.pharms[route_idx][pharm_idx]

        # Include only the sequence members whose types are enabled.
        route_seq = [mem for mem in route_seq if mem.type in self.enabled]

        # Get reactant seqids and features
        if len([mem for mem in route_seq 
                if mem.type == ActionTypes.REACTANT]) > 0:
            rseqids, rids = zip(*[(seqidx, member.idx) 
                                for seqidx, member in enumerate(route_seq)
                                if member.type == ActionTypes.REACTANT])
            rseqids = torch.tensor(rseqids)
            rfeats = torch.tensor(
                    self.routes.rstore.feats[list(rids)]) # list vs tuple
        else:
            rseqids = torch.zeros(0)
            rfeats = torch.zeros(0, self.routes.rstore.feats.shape[1])
        
        # Get product seqids and features
        if len([mem for mem in route_seq 
                if mem.type == ActionTypes.PRODUCT]) > 0:
            pseqids, pids = zip(*[(seqidx, member.idx) 
                                for seqidx, member in enumerate(route_seq)
                                if member.type == ActionTypes.PRODUCT])
            pseqids = torch.tensor(pseqids)
            pfeats = torch.tensor(
                    self.routes.pstore.feats[list(pids)]) # list vs tuple
        else:
            pseqids = torch.zeros(0)
            pfeats = torch.zeros(0, self.routes.pstore.feats.shape[1])

        # Get reaction seqids and features
        if len([mem for mem in route_seq
                if mem.type == ActionTypes.REACTION]) > 0:
            rxnseqids, rxnids = zip(*[(seqidx, member.idx) 
                                    for seqidx, member in enumerate(route_seq)
                                    if member.type == ActionTypes.REACTION])
            rxnseqids = torch.tensor(rxnseqids)
            rxnfeats = torch.tensor(
                    self.routes.rxnstore.feats[list(rxnids)]) # list vs tuple
        else:
            rxnseqids = torch.zeros(0)
            rxnfeats = torch.zeros(0, self.routes.rxnstore.feats.shape[1])
        
        # Get useprod seqids
        if len([mem for mem in route_seq 
                if mem.type == ActionTypes.USEPROD]) > 0:
            uprodseqids, uprodids = zip(*[(seqidx, member.idx) 
                                          for seqidx, member 
                                          in enumerate(route_seq)
                                          if member.type == \
                                            ActionTypes.USEPROD])
            uprodseqids = torch.tensor(uprodseqids)
        else:
            uprodseqids = torch.zeros(0)
        
        # Get start and end seqids
        stseqid = torch.tensor([0])
        endseqid = torch.tensor([len(route_seq) - 1])

        route_data = RouteData(
            rfeats=rfeats,
            pfeats=pfeats,
            rxnfeats=rxnfeats,
            ridx=rseqids,
            pidx=pseqids,
            rxnidx=rxnseqids,
            uprodidx=uprodseqids,
            startidx=stseqid,
            endidx=endseqid
        )

        # Pharmacophore...
        typeids = torch.tensor(pharm.distances.tyidx)
        values = torch.tensor(pharm.distances.dist)
        idx = torch.zeros(typeids.shape[0])

        pharm_data = PharmData(
            typeids=typeids,
            values=values,
            idx=idx
        )

        return SynsploreData(route_data=route_data, 
                             pharm_data=pharm_data)