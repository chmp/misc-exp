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
    
    var FocusCell = widgets.DOMWidgetView.extend({

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

    return {
        FocusCell: FocusCell
    };
});