define(["@jupyter-widgets/base"], function(widgets) {
    function findParentByClass(el, cls) {
        if(el == null) {
            return el;
        }
        if(el.classList.contains(cls)) {
            return el;
        }
        return findParentByClass(el.parentElement, cls);
    }

    const FocusCell = widgets.DOMWidgetView.extend({
        render: function() {
            const button = document.createElement("button");
            button.innerText = "\u2B1A";
            this.el.append(button);

            button.addEventListener("click", () => {
                const cell = findParentByClass(this.el, "cell");

                document.querySelectorAll(".cell").forEach(c => {
                    if(c !== cell) {
                        if(c.style.display == "") {
                            c.style.display = "none";
                        }
                        else {
                            c.style.display = "";
                        }
                     }
                });
            });
        },
    });

    const CommandInput = widgets.DOMWidgetView.extend({
        render: function() {
            const input = document.createElement("input");
            input.style.width = "35em";
            input.value = "";

            input.addEventListener("keydown", (ev) => {
                if(ev.which == 13) {
                    ev.preventDefault();
                    const value = input.value;
                    input.value = "";

                    this.send({type: 'command', value: value});
                }
                else if(ev.which == 27) {
                    ev.preventDefault();
                    input.blur();
                }
            });

            this.el.appendChild(input);
        },
    });

    return {
        FocusCell: FocusCell,
        CommandInput: CommandInput,
    };
});
