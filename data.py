from dataset import GDSDataset
from options import Options
from synthesis import DataFactory
from utils import first_point_to_origin_whole_set
from torch_utils import encode_labels, get_dataloaders

def call_synthesizer(train, n, augm):
    if augm == None:
        return train
    elif augm == "avc":
        return DataFactory.generate_avc(train, n)
    elif augm == "simple":
        return DataFactory.generate_simple(train, n)
    else:
        return DataFactory.generate_chain(train, augm.split("_"), n)

def load_data(options: Options):
    if options.dataset != "gds":
        raise ValueError(f"Not supported yet. Implement {options.dataset} in dataset.py")
    
    # Load data User Dependent (UD) or User Independent (UI)
    if options.condition == "ud":
        dataset = GDSDataset("./", sub_idx=options.sub_idx)
        train, val, test = dataset.ud_split(k=1, fixed=options.fixed)
        print(f"UD - Original train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    elif options.condition == "ui":
        dataset = GDSDataset("./", sub_idx=None)
        train, val, test = dataset.ui_split(split_idx=options.sub_idx, p=options.num_participants, k=options.original_per_class, fixed=options.fixed)
        print(f"UI - Original train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # How many synthetic samples per original sample
    synthetic_per_sample = options.synthetic_per_class // (len(train) // options.num_classes)

    # Use synthetic data for validation
    if options.synthetic_validation:
        test.extend(val) # move validation to test
        val = call_synthesizer(train, n=synthetic_per_sample*2, augm=options.augm)

    # User synthetic data for training
    synthetic_train = call_synthesizer(train, n=synthetic_per_sample, augm=options.augm)
    train.extend(synthetic_train)

    print(f"Synthetic train: {len(train)}, Val: {len(val)}, Real Test: {len(test)}")

    train, val, test = first_point_to_origin_whole_set(train, val, test)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = encode_labels(
        train, val, test
    )

    return get_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, options.batch_size)
