import ipywidgets as widgets
from traitlets import Unicode, validate, List, Union, Any


class HexGame(widgets.DOMWidget):
    _view_name = Unicode('HexGame').tag(sync=True)
    _view_module = Unicode('de.cprohm.hexworld').tag(sync=True)

    def __init__(self, update_callback):
        super().__init__()
        self.update_callback = update_callback
        self.on_msg(self._update)
        self._init_js()

    @classmethod
    def _init_js(cls):
        if getattr(cls, '_js_initialized', False):
            return

        from IPython.display import display, Javascript
        display(Javascript(_hexgame_javascript))
        cls._js_initialized = True

    def _update(self, _, ev, __):
        if ev.get('type') != 'update':
            return

        ev = dict(ev, **self.update_callback(ev))
        self.send(ev)


_hexgame_javascript = '''
require.undef('de.cprohm.hexworld');

define('de.cprohm.hexworld', ["@jupyter-widgets/base"], function(widgets) {
    var c60 = Math.cos(2 * Math.PI * 60 / 360);
    var s60 = Math.sin(2 * Math.PI * 60 / 360);

    return {
        HexGame: widgets.DOMWidgetView.extend({
            render: function() {
                this.canvas = document.createElement('canvas');
                this.canvas.width = 300;
                this.canvas.height = 300;
                this.canvas.tabIndex = 1;

                this.el.appendChild(this.canvas);

                this.model.on('msg:custom', this.onCustom, this);

                this.time = 0;
                this.canvas.addEventListener('click', function() {

                }.bind(this));

                this.canvas.addEventListener('keypress', function(ev) {
                    ev.preventDefault();

                    this.time++;
                    this.send({type: 'update', time: this.time, key: ev.keyCode});
                }.bind(this));
            },

            onCustom: function(ev) {
                // TODO: check ev.time 
                drawGrid(this.canvas.getContext('2d'), ev.grid, {
                    radius: 25,
                    colors: {
                        0: '#ffffff',
                        1: '#333333',
                        2: '#ff0000',
                    },
                });
            },
        }),
    };

    function drawGrid(ctx, grid, options) {
        var radius = options['radius'];
        var colors = options['colors'];

        var sx = 2 * radius * c60  + 5;
        var sy = radius * s60 + 5;

        var dh = 2 * radius * s60;
        var dw = 3 * radius * c60

        for(var i = 0; i < grid.length; ++i) {
            for(var j = 0; j < grid[i].length; ++j) {
                var val = grid[i][j];    
                ctx.fillStyle = colors[val] || '#00ff00';
                hex(ctx, sx + dw * j, sy + i * dh + (j % 2) * radius * s60 , 25);
            }
        }
    }

    function hex(ctx, cx, cy, radius) {
        ctx.beginPath();
        ctx.moveTo(cx - 2 * radius * c60, cy);
        ctx.lineTo(cx - 1 * radius * c60, cy - radius * s60);
        ctx.lineTo(cx + 1 * radius * c60, cy - radius * s60);
        ctx.lineTo(cx + 2 * radius * c60, cy);
        ctx.lineTo(cx + 1 * radius * c60, cy + radius * s60);
        ctx.lineTo(cx - 1 * radius * c60, cy + radius * s60);
        ctx.closePath();
        ctx.stroke();
        ctx.fill();
    }
});
'''
