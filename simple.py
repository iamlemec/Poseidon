# -*- coding: utf-8 -*-
# vispy: gallery 2000
# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
# Author:   Nicolas P .Rougier
# Date:     06/03/2014
# Abstract: GPU computing usingthe framebuffer
# Keywords: framebuffer, GPU computing
# -----------------------------------------------------------------------------

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
uniform sampler2D texture;
varying vec2 v_texcoord;

void main()
{
    float v = texture2D(texture, v_texcoord).r;
    gl_FragColor = vec4(v, v, v, 1.0);
}
"""

compute_vertex = """
attribute vec2 position;
attribute vec2 texcoord;
varying vec2 v_texcoord;

void main()
{
    gl_Position = vec4(position, 0.0, 1.0);
    v_texcoord = texcoord;
}
"""

compute_fragment = """
uniform sampler2D texture;
uniform float dx;
uniform float dy;
uniform float width;
uniform float steep;
varying vec2 v_texcoord;

float sigmoid(float x)
{
    return 1.0/(1.0+exp(-2.0*x));
}

void main(void)
{
    vec2 p = v_texcoord;

    float c = texture2D(texture, p).r;
    float n =
        texture2D(texture, p+vec2(-dx,-dy)).r
      + texture2D(texture, p+vec2(0.0,-dy)).r
      + texture2D(texture, p+vec2(dx,-dy)).r
      + texture2D(texture, p+vec2(-dx,0.0)).r
      + texture2D(texture, p+vec2(dx,0.0)).r
      + texture2D(texture, p+vec2(-dx,dy)).r
      + texture2D(texture, p+vec2(0.0,dy)).r
      + texture2D(texture, p+vec2(dx,dy)).r;

    float u =
        sigmoid(steep*(c-0.5)) * ( 1.0 - sigmoid(steep*((2.0-width)-n)) - sigmoid(steep*(n-(3.0+width))) )
      + sigmoid(steep*(0.5-c)) * ( 1.0 - sigmoid(steep*((3.0-width)-n)) - sigmoid(steep*(n-(3.0+width))) );

    gl_FragColor = vec4(u, 0, 0, 0);
}
"""

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, title='Poseidon', size=(512, 512), keys='interactive')

        # program state
        self.run = True
        self.speed = 1
        self.width = 0.5
        self.steep = 10.0

        # init field
        comp_w, comp_h = self.comp_size = self.size
        state0 = np.random.uniform(size=(comp_w, comp_h, 4)).astype(np.float32)
        texture = gloo.Texture2D(state0, wrapping='repeat', interpolation='linear')

        # common coordinates
        position = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        texcoord = [(0, 0), (0, 1), (1, 0), (1, 1)]

        # compute logic
        self.compute = gloo.Program(compute_vertex, compute_fragment, 4)
        self.compute['texture'] = texture
        self.compute['position'] = position
        self.compute['texcoord'] = texcoord
        self.compute['dx'] = 1.0 / comp_w
        self.compute['dy'] = 1.0 / comp_h
        self.compute['width'] = self.width
        self.compute['steep'] = self.steep

        # render logic
        self.render = gloo.Program(render_vertex, render_fragment, 4)
        self.render['position'] = position
        self.render['texcoord'] = texcoord
        self.render['texture'] = texture

        # hook output of compute up to texture
        self.fbo = gloo.FrameBuffer(texture, gloo.RenderBuffer(self.comp_size))

        # scene options
        gloo.set_state(depth_test=False, clear_color='black')

        # some kind of event loop
        self._timer = app.Timer('auto', connect=self.update, start=True)
        self.show()

    def on_draw(self, event):
        if self.run:
            with self.fbo:
                gloo.set_viewport(0, 0, *self.comp_size)
                for _ in range(self.speed):
                    self.compute.draw('triangle_strip')
        gloo.clear(color=True)
        gloo.set_viewport(0, 0, *self.physical_size)
        self.render.draw('triangle_strip')

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *self.physical_size)

    def on_key_press(self, event):
        key = event.key.name
        if key == 'Space':
            self.run = not self.run
        elif key == 'Z':
            self.speed = min(20, self.speed+1)
            print(f'speed = {self.speed}')
        elif key == 'X':
            self.speed = max(1, self.speed-1)
            print(f'speed = {self.speed}')
        elif key == 'O':
            self.width *= 1.2
            self.compute['width'] = self.width
            print(f'width = {self.width}')
        elif key == 'P':
            self.width /= 1.2
            self.compute['width'] = self.width
            print(f'width = {self.width}')
        elif key == 'K':
            self.steep *= 1.2
            self.compute['steep'] = self.steep
            print(f'steep = {self.steep}')
        elif key == 'L':
            self.steep /= 1.2
            self.compute['steep'] = self.steep
            print(f'steep = {self.steep}')

if __name__ == '__main__':
    canvas = Canvas()
    app.run()
