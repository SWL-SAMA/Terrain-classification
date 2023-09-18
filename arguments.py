import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--run_mode', type=str, default='test', help="Run this program to train or test")
parser.add_argument('--train_epochs', type=int, default=50, help='Set training epochs')
args = parser.parse_args()