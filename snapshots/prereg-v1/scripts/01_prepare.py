import argparse
import yaml
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders import load_d1, load_d2
from src.detect import detect_events
from src.injector import inject_spikes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--stage', required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    if args.stage == 'load':
        if args.dataset == 'D1':
            load_d1(cfg)
        elif args.dataset == 'D2':
            load_d2(cfg)
    elif args.stage == 'events':
        detect_events(cfg, args.dataset)
    elif args.stage == 'inject':
        inject_spikes(cfg, args.dataset)
    elif args.stage == 'all':
        if args.dataset == 'D1':
            load_d1(cfg)
        elif args.dataset == 'D2':
            load_d2(cfg)
        detect_events(cfg, args.dataset)
        inject_spikes(cfg, args.dataset)

if __name__ == '__main__':
    main()
