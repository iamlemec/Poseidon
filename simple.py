# poseidon

import argparse
import pytoml as toml
import numpy as np
from vispy import app, gloo

render_vertex = """
attribute vec2 position;
attribute vec2 texcoord;
varying vec2 v_texcoord;

void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    v_texcoord = texcoord;
}
"""

render_fragment = """
uniform sampler2D cells;
varying vec2 v_texcoord;

void main()
{
    float v = texture2D(cells, v_texcoord).r;
    gl_FragColor = vec4(v, v, v, 1.0);
}
"""

cells_vertex = """
attribute vec2 position;
attribute vec2 texcoord;
varying vec2 v_texcoord;

void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    v_texcoord = texcoord;
}
"""

cells_fragment = """
uniform sampler2D cells;
uniform sampler2D field;
uniform float dx;
uniform float dy;
uniform float rule_width;
uniform float rule_steep;
uniform float couple_cells;
uniform float couple_field;
varying vec2 v_texcoord;

float sigmoid(float x)
{
    return 1.0/(1.0+exp(-2.0*x));
}

void main(void)
{
    vec2 p = v_texcoord;

    float c = texture2D(cells, p).r;
    float n =
        texture2D(cells, p+vec2(-dx,-dy)).r
      + texture2D(cells, p+vec2(0.0,-dy)).r
      + texture2D(cells, p+vec2(dx,-dy)).r
      + texture2D(cells, p+vec2(-dx,0.0)).r
      + texture2D(cells, p+vec2(dx,0.0)).r
      + texture2D(cells, p+vec2(-dx,dy)).r
      + texture2D(cells, p+vec2(0.0,dy)).r
      + texture2D(cells, p+vec2(dx,dy)).r;

    float f = texture2D(field, p).r;

    float v =
        sigmoid(rule_steep*(c-0.5)) * ( 1.0
            - sigmoid(rule_steep*((2.0-rule_width)-n))
            - sigmoid(rule_steep*(n-(3.0+rule_width)))
        )
        + sigmoid(rule_steep*(0.5-c)) * ( 1.0
            - sigmoid(rule_steep*((3.0-rule_width)-n))
            - sigmoid(rule_steep*(n-(3.0+rule_width)))
        );

    float u = ( v + couple_field * f ) * couple_cells;

    gl_FragColor = vec4(u, 0, 0, 0);
}
"""

field_vertex = """
attribute vec2 position;
attribute vec2 texcoord;
varying vec2 v_texcoord;

void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    v_texcoord = texcoord;
}
"""

field_fragment = """
uniform int field_type;
varying vec2 v_texcoord;

void main(void)
{
    vec2 p = v_texcoord;

    float rad = 0.3;
    float cur = 1.5;
    vec2 fzero = vec2(0.5, 0.5);

    float u;
    if (field_type == 0) {
        u = 0.0;
    } else if (field_type == 1) {
        float f0 = pow(pow(abs(p.x-fzero.x)/rad, cur) + pow(abs(p.y-fzero.y)/rad, cur), 1.0/cur);
        u = max(0.0, 1.0-f0);
    }

    gl_FragColor = vec4(u, 0, 0, 0);
}
"""

class Canvas(app.Canvas):
    def __init__(self, **config):
        app.Canvas.__init__(self, size=config['grid_size'], title='Poseidon', keys='interactive')

        # program state
        self.run = True
        self.step = False
        self.speed = config['speed']
        self.rule_width = config['rule_width']
        self.rule_steep = config['rule_steep']
        self.couple_cells = config['couple_cells']
        self.couple_field = config['couple_field']
        self.field_type = config['field_type']

        # init cells and field
        comp_w, comp_h = self.comp_size = config['grid_size']
        field0 = np.zeros((comp_w, comp_h, 4), dtype=np.float32)
        cells0 = np.random.uniform(size=(comp_w, comp_h, 4)).astype(np.float32)
        cells = gloo.Texture2D(cells0, wrapping='repeat', interpolation='linear')
        field = gloo.Texture2D(field0, wrapping='clamp_to_edge', interpolation='linear')

        # common coordinates
        position = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        texcoord = [(0, 0), (0, 1), (1, 0), (1, 1)]

        # compute logic
        self.prog_cells = gloo.Program(cells_vertex, cells_fragment, 4)
        self.prog_cells['position'] = position
        self.prog_cells['texcoord'] = texcoord
        self.prog_cells['cells'] = cells
        self.prog_cells['field'] = field
        self.prog_cells['dx'] = 1.0 / comp_w
        self.prog_cells['dy'] = 1.0 / comp_h
        self.prog_cells['rule_width'] = self.rule_width
        self.prog_cells['rule_steep'] = self.rule_steep
        self.prog_cells['couple_cells'] = self.couple_cells
        self.prog_cells['couple_field'] = self.couple_field

        # field logic
        self.prog_field = gloo.Program(field_vertex, field_fragment, 4)
        self.prog_field['position'] = position
        self.prog_field['texcoord'] = texcoord
        self.prog_field['field_type'] = self.field_type

        # render logic
        self.prog_render = gloo.Program(render_vertex, render_fragment, 4)
        self.prog_render['position'] = position
        self.prog_render['texcoord'] = texcoord
        self.prog_render['cells'] = cells

        # hook output of compute up to texture
        self.fbo_cells = gloo.FrameBuffer(cells, gloo.RenderBuffer(self.comp_size))
        self.fbo_field = gloo.FrameBuffer(field, gloo.RenderBuffer(self.comp_size))

        # scene options
        gloo.set_state(depth_test=False, clear_color='black')

        # some kind of event loop
        self._timer = app.Timer('auto', connect=self.update, start=True)
        self.show()

    def on_draw(self, event):
        if self.run or self.step:
            gloo.set_viewport(0, 0, *self.comp_size)
            for _ in range(1 if self.step else self.speed):
                with self.fbo_field:
                    self.prog_field.draw('triangle_strip')
                with self.fbo_cells:
                    self.prog_cells.draw('triangle_strip')
            if self.step:
                self.step = False
        gloo.clear(color=True)
        gloo.set_viewport(0, 0, *self.physical_size)
        self.prog_render.draw('triangle_strip')

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *self.physical_size)

    def on_key_press(self, event):
        key = event.key.name
        if key == 'Space':
            self.run = not self.run
            print(f'run = {self.run}')
        elif key == 'F':
            self.step = True
            print('step')
        elif key == 'Z':
            self.speed = min(20, self.speed+1)
            print(f'speed = {self.speed}')
        elif key == 'X':
            self.speed = max(1, self.speed-1)
            print(f'speed = {self.speed}')
        elif key == 'O':
            self.rule_width *= 1.2
            self.prog_cells['rule_width'] = self.rule_width
            print(f'rule_width = {self.rule_width}')
        elif key == 'P':
            self.rule_width /= 1.2
            self.prog_cells['rule_width'] = self.rule_width
            print(f'rule_width = {self.rule_width}')
        elif key == 'K':
            self.rule_steep *= 1.2
            self.prog_cells['rule_steep'] = self.rule_steep
            print(f'rule_steep = {self.rule_steep}')
        elif key == 'L':
            self.rule_steep /= 1.2
            self.prog_cells['rule_steep'] = self.rule_steep
            print(f'rule_steep = {self.rule_steep}')
        elif key == 'T':
            self.field_type = (self.field_type + 1) % 2
            self.prog_field['field_type'] = self.field_type
            print(f'field_type = {self.field_type}')
        elif key == 'Q':
            self.couple_field *= 1.2;
            self.prog_cells['couple_field'] = self.couple_field
            print(f'couple_field = {self.couple_field}')
        elif key == 'W':
            self.couple_field /= 1.2;
            self.prog_cells['couple_field'] = self.couple_field
            print(f'couple_field = {self.couple_field}')
        elif key == 'A':
            self.couple_cells *= 1.01;
            self.prog_cells['couple_cells'] = self.couple_cells
            print(f'couple_cells = {self.couple_cells}')
        elif key == 'S':
            self.couple_cells /= 1.01;
            self.prog_cells['couple_cells'] = self.couple_cells
            print(f'couple_cells = {self.couple_cells}')

# parse command line arguments
parser = argparse.ArgumentParser(description='General celular automata simulator in CUDA.')
parser.add_argument('--config', type=str, default=None, help='Config file from which to load rules')
parser.add_argument('--rule', type=str, default=None, help='Rule name to load from config file')
parser.add_argument('--run', type=bool, default=None, help='Start the simulation running')
parser.add_argument('--couple-cells', type=float, default=None, help='Stength of self coupling')
parser.add_argument('--couple-field', type=float, default=None, help='Strength of field coupling')
parser.add_argument('--rule-width', type=float, default=None, help='Width of automata rule (from GOL)')
parser.add_argument('--rule-steep', type=float, default=None, help='Steepness of automata rule')
parser.add_argument('--field-type', type=str, default=None, help='Values: null, radial, coupled')
parser.add_argument('--grid-size', type=str, default=None, help='Width and height of cell grid')
parser.add_argument('--speed', type=int, default=None, help='speed at which to run (higher is faster)')
args = parser.parse_args()

# base default
config = {
    'run': True,
    'rule_width': 0.5,
    'rule_steep': 3.0,
    'couple_cells': 0.92,
    'couple_field': 1.0,
    'field_type': 'null',
    'grid_size': '512x512',
    'speed': 1
}

# if config file specified
if args.config is not None:
    path_conf = args.__dict__.pop('config')
    rule_name = args.__dict__.pop('rule')
    config_file = toml.load(open(path_conf))
    config_rule = config_file[rule_name]
    config = dict(config, **config_rule)

# process manual overrides
config_args = {k: v for k, v in args.__dict__.items() if v is not None}
config = dict(config, **config_args)

# process grid size
if 'grid_size' in config:
    grid_x, grid_y = map(int, config['grid_size'].split('x'))
    config['grid_size'] = grid_x, grid_y

# process field type
field_map = {
    'null': 0,
    'radial': 1,
    # 'coupled': 2,
    # 'swizzled': 3
}
config['field_type'] = field_map[config['field_type']]

# create canvas
canvas = Canvas(**config)

# run app
app.run()
