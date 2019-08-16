# ising

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
uniform sampler2D chaos;
uniform float dx;
uniform float dy;
uniform float field;
uniform float temp;
varying vec2 v_texcoord;

float sigmoid(float x)
{
    return 1.0/(1.0+exp(-2.0*x));
}

float sign(float x) {
    return 2.0*step(0.0, x) - 1.0;
}

void main(void)
{
    vec2 p = v_texcoord;

    float c = texture2D(cells, p).r;
    float n = texture2D(cells, p+vec2(0.0, dy)).r + texture2D(cells, p+vec2( dx,0.0)).r
            + texture2D(cells, p+vec2(0.0,-dy)).r + texture2D(cells, p+vec2(-dx,0.0)).r;

    float r = texture2D(chaos, p).r;

    float e = n - field*c;
    float z = sigmoid(e/temp);
    float f = sign(z-r);
    float u = f*c;

    gl_FragColor = vec4(u, 0, 0, 0);
}
"""

chaos_vertex = """
attribute vec2 position;
attribute vec2 texcoord;
varying vec2 v_texcoord;

void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    v_texcoord = texcoord;
}
"""

chaos_fragment = """
uniform sampler2D chaos;
varying vec2 v_texcoord;

float rand(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898,78.233))) * 43758.5453);
}

void main(void)
{
    vec2 p = v_texcoord;

    float c = texture2D(chaos, p).r;

    float u = rand(c*p);

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
        self.field = config['field']
        self.temp = config['temp']

        # init cells and field
        comp_w, comp_h = self.comp_size = config['grid_size']
        cells0 = np.random.uniform(size=(comp_w, comp_h, 4)).astype(np.float32)
        chaos0 = np.random.uniform(size=(comp_w, comp_h, 4)).astype(np.float32)
        cells = gloo.Texture2D(cells0, wrapping='repeat', interpolation='linear')
        chaos = gloo.Texture2D(chaos0, wrapping='repeat', interpolation='linear')

        # common coordinates
        position = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        texcoord = [(0, 0), (0, 1), (1, 0), (1, 1)]

        # compute logic
        self.prog_cells = gloo.Program(cells_vertex, cells_fragment, 4)
        self.prog_cells['position'] = position
        self.prog_cells['texcoord'] = texcoord
        self.prog_cells['cells'] = cells
        self.prog_cells['chaos'] = chaos
        self.prog_cells['dx'] = 1.0 / comp_w
        self.prog_cells['dy'] = 1.0 / comp_h
        self.prog_cells['field'] = self.field
        self.prog_cells['temp'] = self.temp

        # chaos logic
        self.prog_chaos = gloo.Program(chaos_vertex, chaos_fragment, 4)
        self.prog_chaos['position'] = position
        self.prog_chaos['texcoord'] = texcoord
        self.prog_chaos['chaos'] = chaos

        # render logic
        self.prog_render = gloo.Program(render_vertex, render_fragment, 4)
        self.prog_render['position'] = position
        self.prog_render['texcoord'] = texcoord
        self.prog_render['cells'] = cells

        # hook output of compute up to texture
        self.fbo_cells = gloo.FrameBuffer(cells, gloo.RenderBuffer(self.comp_size))
        self.fbo_chaos = gloo.FrameBuffer(chaos, gloo.RenderBuffer(self.comp_size))

        # scene options
        gloo.set_state(depth_test=False, clear_color='black')

        # some kind of event loop
        self._timer = app.Timer('auto', connect=self.update, start=True)
        self.show()

    def on_draw(self, event):
        if self.run or self.step:
            gloo.set_viewport(0, 0, *self.comp_size)
            for _ in range(1 if self.step else self.speed):
                with self.fbo_chaos:
                    self.prog_chaos.draw('triangle_strip')
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

# parse command line arguments
parser = argparse.ArgumentParser(description='General celular automata simulator in CUDA.')
parser.add_argument('--config', type=str, default=None, help='Config file from which to load rules')
parser.add_argument('--rule', type=str, default=None, help='Rule name to load from config file')
parser.add_argument('--run', type=bool, default=None, help='Start the simulation running')
parser.add_argument('--grid-size', type=str, default=None, help='Width and height of cell grid')
parser.add_argument('--speed', type=int, default=None, help='speed at which to run (higher is faster)')
parser.add_argument('--field', type=float, default=None, help='external field strength')
parser.add_argument('--temp', type=float, default=None, help='external temperature')
args = parser.parse_args()

# base default
config = {
    'run': True,
    'grid_size': '512x512',
    'speed': 1,
    'field': 0.5,
    'temp': 1.0
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

# create canvas
canvas = Canvas(**config)

# run app
app.run()
