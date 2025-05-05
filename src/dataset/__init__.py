from src.dataset.crud_datasets import CRUDDataset
from src.dataset.hotpop_datasets import HOTPOPDataset
from src.dataset.musique_datasets import MUSIQUEDataset
from src.dataset.wiki_datasets import WIKIDataset

load_dataset = {
    'CRUD': CRUDDataset,
    'HOTPOPQA':HOTPOPDataset,
    'MUSIQUE':MUSIQUEDataset,
    'WIKI':WIKIDataset
}