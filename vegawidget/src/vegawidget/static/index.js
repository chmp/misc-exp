// piggy-back of the ipyvega extension (https://github.com/vega/ipyvega/blob/master/src/index.ts)
define(
    'vegawidget',
    ["@jupyter-widgets/base", 'nbextensions/jupyter-vega/index'],
    (widgets, {vegaEmbed}) => {

    const VegaWidget = widgets.DOMWidgetView.extend({
        render: function() {
            const reembed = () => {
                this.view = null;
                const spec = JSON.parse(this.model.get('spec_source'));

                if(spec == null) {
                    return
                }

                vegaEmbed(this.el, spec)
                .then(({view}) => {
                    this.view = view;
                })
                .catch(err => console.error(err));
            };

            this.model.on('change:spec_source', reembed);
            this.model.on('msg:custom', ev => {
                if(ev.type != 'update') {
                    return;
                }
                if(this.view == null) {
                    console.error('no view attached to widget');
                    return
                }

                const filter = new Function('datum', 'return (' + (ev.remove || 'false') + ')');
                const newValues = ev.insert || [];

                const changeSet = (
                    // TODO: is this supported?
                    this.view.changeset()
                    .insert(newValues)
                    .remove(filter)
                );
                this.view.change(ev.key, changeSet).run();
            });

            // initial rendering
            reembed()
        },
    });

    return {VegaWidget};
});