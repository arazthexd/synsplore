from typing import List
import random

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tensordict import TensorDict

class SynsploreDataset(Dataset):
    """Pytorch dataset class combining synthesis routes and pharmacophore data.

    This dataset class combines the synthesis data and pharmacophore data into
    a single data object. It uses `TensorDict` as its output, both for individual
    synthesis routes and pharmacophore data, and the combined data object.

    Either when using the `__getitem__` method or indexing (e.g. dataset[0]), 
    or when several data are collated into a batch (e.g. in data loader or
    when using the `collate_fn` method), the output will have two main fields:
    - `syndata`: The synthesis route data. A `TensorDict` itself.
    - `pharmdata`: The pharmacophore data. A `TensorDict` itself.

    The synthesis route data contains:
    - `rfeats`: The features of reactants in the synthesis route. It always has
        two dimensions, and has a shape of (n_reactants, n_feats). When the
        route data is batched, it will remain its two dimensions and the concat
        will happen on the first dimension.
    - `pfeats`: The features of products in the synthesis route. Its shape is
        similar to `rfeats`. If `distable_products` is True, the first dimension
        of `pfeats` will a size of 0.
    - `rxnfeats`: The features of reactions in the synthesis route. The shape
        is similar to the previous two fields.
    - `ridx`: The sequence indices of reactants in the synthesis route. If the
        route data is not batched, it will have a shape of (n_reactants,). Each
        value is the index of the reactant in the synthesis route sequence. If
        batched, it will have a shape of (n_reactants, 2). In this case, each
        row will have two values, the first one showing the index of the
        synthesis route in the batch, and the second one showing the index of
        the reactant in the synthesis route's sequence.
    - `pidx`: The sequence indices of products in the synthesis route. The shape
        is similar to `ridx`.
    - `rxnidx`: The sequence indices of reactions in the synthesis route. The
        shape is similar to `ridx`.
    - `usepidx`: The sequence indices of the "use_last_product" action in the 
        synthesis route. The shape is similar to `ridx`. However, unlike the
        previous index fields, this one does not have a separate "feat" field
        in the tensor dict, as it potentially needs to be embedded and is simply
        an action rather than a specific featurized member of the sequence.
    - `stidx`: The sequence index of the "start" action in the synthesis route.
        This will always be one dimensional and its size would be equal to the
        batch size or 1 if the route data is not batched.
    - `endidx`: The sequence index of the "end" action in the synthesis route.
        Its shape will be similar to `stidx`.

    The pharmacophore data contains:
    - `tys`: The feature type ids of pharmacophoric features in the 
        combinations. This will have a shape of 
        (n_combinations, n_pairs, 2) if the data is not batched. For every 
        pharmacophore model/profile, there will be several combinations
        of the features. Each combination will have a number of pairs between
        features in the combination (e.g. if considering a 4-feature 
        combination, there will be 6 pairs). Each pair will have two feature
        type ids. If the data is batched, the shape will be
        (batch_size, max(n_combinations), n_pairs, 2). During batching, the
        first dimension will be padded on the left size with `-1`s.
    - `tyids`: The combination type ids. This will assign one id for every
        unique combination of pharmacophoric features. It will have a shape of
        (n_combinations,) if the data is not batched. If the data is batched,
        the shape will be (batch_size, max(n_combinations)). Its batching is
        similar to `tys`.
    - `dists`: The distances of pharmacophoric features in the combinations.
        This will have a shape of (n_combinations, n_pairs) if the data is not
        batched. If the data is batched, the shape will be
        (batch_size, max(n_combinations), n_pairs). Batching similar to `tys`.
    - `angles`: The angles (in radians) between the pharmacophoric feature 
        direcion of the first and second features in the combination, with the 
        different vector between their coordinates. This will have a shape of
        (n_combinations, n_pairs, 2) if the data is not batched. Otherwise,
        it will have a shape of (batch_size, max(n_combinations), n_pairs, 2).
        Batching similar to `tys`.

    Attributes:
        num_pharm_ty (int): The number of pharmacophore feature types. (e.g. 
            hbond-hyd, aromatic, etc.)
        num_pharm_tyid (int): The maximum number of pharmacophore combination 
            types. For example, If 2-feature combinations is used, this would 
            be equal to:
              num_pharm_ty * (num_pharm_ty-1) / 2 + num_pharm_ty
        disable_products (bool): Whether to exclude products in the synthesis 
            route output data or not.
    """

    def __init__(self, disable_products: bool = True):
        """Initiates SynsploreDataset class.

        Args:
            disable_products (bool, optional): Whether to exclude products 
                in the synthesis route output data or not. Defaults to True.
        """
        self.num_pharm_ty = 6
        self.num_pharm_tyid = 15
        self.disable_products: bool = disable_products

    def __getitem__(self, idx: int) -> TensorDict:
        pass

    def random_data(self) -> TensorDict:
        l = random.randint(0, 100)
        pharmdata = TensorDict(
            {
                "tys": torch.randint(0, self.num_pharm_ty, size=(l, 1, 2)),
                "tyids": torch.randint(0, self.num_pharm_tyid, size=(l,)),
                "dists": torch.randn((l, 1)) * 5,
                "angles": torch.clamp(torch.randn((l, 1, 2)) * 2, 0, torch.pi)
            },
            batch_size=l
        )
        
        route = ["st"]
        for i in range(random.randint(1, 4)):
            for j in range(random.randint(1, 4)):
                route.append("r")
            route.append("rxn")
            if not self.disable_products:
                route.append("p")
        route.append("end")
        
        while True:
            unusedp_count = 0
            for i, mem in enumerate(list(route)):
                if mem == "rxn":
                    unusedp_count += 1
                elif mem == "r":
                    uptemp = 0
                    for memtemp in list(route[i:]):
                        if memtemp == "rxn":
                            break
                        if memtemp == "usep":
                            uptemp += 1

                    if unusedp_count - uptemp > 0:
                        route[i] = ["r", "usep"][random.randint(0, 1)]
                        if route[i] == "usep":
                            unusedp_count -= 1
                elif mem == "usep":
                    unusedp_count -= 1
            
            if unusedp_count == 1:
                break
            else:
                # print(route, unusedp_count)
                pass
            
        lr = route.count("r")
        lrxn = route.count("rxn")
        lp = route.count("p")

        rfeats = torch.randint(0, 2, (lr, 1024))
        pfeats = torch.randint(0, 2, (lp, 1024))
        rxnfeats = torch.zeros((lrxn, 27), dtype=int)
        rxnfeats[torch.arange(rxnfeats.shape[0]), 
                 [random.randint(0, 26) for _ in range(rxnfeats.shape[0])]] = 1
        ridx = torch.tensor([i for i, mem in enumerate(route) 
                             if mem == "r"], dtype=torch.long)
        pidx = torch.tensor([i for i, mem in enumerate(route) 
                             if mem == "p"], dtype=torch.long)
        rxnidx = torch.tensor([i for i, mem in enumerate(route) 
                               if mem == "rxn"], dtype=torch.long)
        usepidx = torch.tensor([i for i, mem in enumerate(route) 
                                if mem == "usep"], dtype=torch.long)
        stidx = torch.tensor([0], dtype=torch.long)
        endidx = torch.tensor([len(route)-1], dtype=torch.long)
        syndata = TensorDict(
            {
                "rfeats": rfeats,
                "pfeats": pfeats,
                "rxnfeats": rxnfeats,
                "ridx": ridx,
                "pidx": pidx,
                "rxnidx": rxnidx,
                "usepidx": usepidx,
                "stidx": stidx,
                "endidx": endidx
            }
        )

        return TensorDict({"syndata": syndata, "pharmdata": pharmdata})

    @staticmethod
    def collate_syndata(sdlist: List[TensorDict]) -> TensorDict:
        rfeats = torch.cat([syndata["rfeats"] for syndata in sdlist], dim=0)
        pfeats = torch.cat([syndata["pfeats"] for syndata in sdlist], dim=0)
        rxnfeats = torch.cat([syndata["rxnfeats"] for syndata in sdlist], dim=0)

        ridx_list: List[torch.Tensor] = [
            torch.stack([torch.ones_like(syndata["ridx"])*i, 
                         syndata["ridx"]], dim=-1) 
            for i, syndata in enumerate(sdlist)
        ]
        ridx = torch.cat(ridx_list, dim=0)
        
        pidx_list: List[torch.Tensor] = [
            torch.stack([torch.ones_like(syndata["pidx"])*i, 
                         syndata["pidx"]], dim=-1) 
            for i, syndata in enumerate(sdlist)
        ]
        pidx = torch.cat(pidx_list, dim=0)
        
        rxnidx_list: List[torch.Tensor] = [
            torch.stack([torch.ones_like(syndata["rxnidx"])*i, 
                         syndata["rxnidx"]], dim=-1) 
            for i, syndata in enumerate(sdlist)
        ]
        rxnidx = torch.cat(rxnidx_list, dim=0)

        usepidx_list: List[torch.Tensor] = [
            torch.stack([torch.ones_like(syndata["usepidx"])*i, 
                         syndata["usepidx"]], dim=-1) 
            for i, syndata in enumerate(sdlist)
        ]
        usepidx = torch.cat(usepidx_list, dim=0)

        stidx = torch.tensor([syndata["stidx"] for syndata in sdlist])
        endidx = torch.tensor([syndata["endidx"] for syndata in sdlist])
        
        return TensorDict({
            "rfeats": rfeats,
            "pfeats": pfeats,
            "rxnfeats": rxnfeats,
            "ridx": ridx,
            "pidx": pidx,
            "rxnidx": rxnidx,
            "usepidx": usepidx,
            "stidx": stidx,
            "endidx": endidx
        })
    
    @staticmethod
    def collate_pharmdata(pdlist: List[TensorDict]) -> TensorDict:
        tys_list = [pharmdata["tys"] for pharmdata in pdlist]
        tyids_list = [pharmdata["tyids"] for pharmdata in pdlist]
        dists_list = [pharmdata["dists"] for pharmdata in pdlist]
        angles_list = [pharmdata["angles"] for pharmdata in pdlist]
        
        tys = pad_sequence(tys_list, batch_first=True, padding_value=-1)
        tyids = pad_sequence(tyids_list, batch_first=True, padding_value=-1)
        dists = pad_sequence(dists_list, batch_first=True, padding_value=-1)
        angles = pad_sequence(angles_list, batch_first=True, padding_value=-1)

        return TensorDict({
            "tys": tys,
            "tyids": tyids,
            "dists": dists,
            "angles": angles
        }, batch_size=tyids.shape[0])
    
    @staticmethod
    def collate_fn(dlist: List[TensorDict]):
        return TensorDict({
            "syndata": SynsploreDataset.collate_syndata([
                data["syndata"] for data in dlist
            ]),
            "pharmdata": SynsploreDataset.collate_pharmdata([
                data["pharmdata"] for data in dlist
            ])
        })
        
if __name__ == "__main__":
    print(SynsploreDataset()[0])
    print(SynsploreDataset.collate_fn([SynsploreDataset().random_data() 
                                       for _ in range(100)]))