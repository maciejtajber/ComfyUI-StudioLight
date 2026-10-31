import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "StudioLight.CustomColorPicker",
    async getNodeCustomMenuOptions(node, options) {
        options.push({
            content: "🎨 Set Custom Color...",
            callback: () => {
                let input = document.getElementById("studiolight-color-picker");
                if (!input) {
                    input = document.createElement("input");
                    input.id = "studiolight-color-picker";
                    input.type = "color";
                    input.style.display = "none";
                    document.body.appendChild(input);
                }
                input.value = node.color || "#4a2840";
                input.oninput = (e) => {
                    const hex = e.target.value;
                    node.color = hex;
                    node.bgcolor = hex;
                    app.graph.setDirtyCanvas(true, true);
                };
                input.click();
            }
        });
    }
});
