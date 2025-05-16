from .dataset import SynsploreDataset

def _get_random_input(batch_size=100):
    return SynsploreDataset.collate_fn(
        [SynsploreDataset().random_data() for _ in range(100)]
    )