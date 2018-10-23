import os
import argparse
import pytoml as toml

parser = argparse.ArgumentParser(description='General celular automata simulator in CUDA.')
parser.add_argument('--config', type=str, default=None, help='Config file from which to load rules')
parser.add_argument('--rule', type=str, default='default', help='Rule name to load from config file')
parser.add_argument('--self-couple', type=float, default=None, help='Stength of self coupling')
parser.add_argument('--field-couple', type=float, default=None, help='Strength of field coupling')
parser.add_argument('--field-type', type=str, default=None, help='Values: null, radial, coupled')
parser.add_argument('--rule-width', type=float, default=None, help='Width of automata rule (from GOL)')
parser.add_argument('--rule-steep', type=float, default=None, help='Steepness of automata rule')
parser.add_argument('--grid-size', type=str, default=None, help='Width and height of cell grid')
parser.add_argument('--speed', type=float, default=None, help='speed at which to run (lower is faster)')
args = parser.parse_args()

# base default
config = {
    'self_couple': 0.92,
    'field_couple': 1.0,
    'field_type': 'null',
    'rule_width': 0.5,
    'rule_steep': 3.0,
    'grid_size': '400x400',
    'speed': 1.0
}

# if config file specified
if args.config is not None:
    config_file = toml.load(open(args.config))
    config_rule = config_file[args.rule]
    config = dict(config, **config_rule)

# process manual overrides
config_args = {k: v for k, v in args.__dict__.items() if v is not None}
config = dict(config, **config_args)

# convert to true parameters
field_map = {
    'null': 0,
    'radial': 1,
    'coupled': 2,
    'swizzled': 3
}
process_size = lambda size:  [s.strip() for s in size.lower().split('x')]
config['field_type'] = field_map[config['field_type']]
config['grid_width'], config['grid_height'] = process_size(config.pop('grid_size'))
config['speed'] = 10/config['speed']

# execute proper command
command = """./Poseidon {self_couple} {field_couple} {rule_width} {rule_steep} {field_type} {grid_width} {grid_height} {speed}""".format(**config)
print(command)
os.system(command)
