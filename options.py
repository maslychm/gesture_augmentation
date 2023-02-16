import argparse
from pprint import pformat
from sys import platform

class Options:
    """Parses and stores the command line arguments."""
    suppored_datasets = ["gds"]
    dataset_classes = {
        "gds": 16
    }

    batch_size = 512
    learning_rate = 0.001

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Testing pipeline")
        
        self.parser.add_argument("--dataset", type=str, default="gds", choices=Options.suppored_datasets)

        self.parser.add_argument("--condition", type=str, default="ud", choices=["ud", "ui"])

        self.parser.add_argument("--sub_idx", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        self.parser.add_argument("--num_participants", type=int, default=1, help="Num. participants for the UI condition")

        self.parser.add_argument("--original_per_class", type=int, default=1, help="Num. original samples per class")

        self.parser.add_argument("--synthetic_per_class", type=int, default=300, help="Num. synthetic samples per class")

        self.parser.add_argument("--augm", type=str, default="None")

        self.parser.add_argument("--synthetic_validation", type=str, default="True", choices=["True", "False"], help="Use synthetic data for validation?")

        self.parser.add_argument("--fixed", type=str, default="True", choices=["True", "False"], help="Use fixed seed?")

        self.fixed = None
        self.dataset = None
        self.condition = None
        self.sub_idx = None
        self.num_participants = None
        self.original_per_class = None
        self.synthetic_per_class = None
        self.augm = None
        self.synthetic_validation = None

        self.platform = platform

    def parse(self, use_defaults=False):
        """Parse command line arguments."""
        if use_defaults:
            args = self.parser.parse_args([])
        else:
            args = self.parser.parse_args()

        self.fixed = eval(args.fixed)
        self.dataset = args.dataset
        self.condition = args.condition
        self.sub_idx = args.sub_idx
        self.num_participants = args.num_participants
        self.original_per_class = args.original_per_class
        self.synthetic_per_class = args.synthetic_per_class
        self.augm = args.augm
        self.synthetic_validation = eval(args.synthetic_validation)

        if self.augm == "None":
            self.augm = None

        self.num_classes = Options.dataset_classes[self.dataset]

        return self
    
    def __str__(self):
        """Prints the options."""
        return pformat(vars(self))
    


        


