/**
 * ComfyUI-StudioLight — Live Preset Synchronizer
 * Author: Maciej Tajber (https://github.com/maciejtajber)
 * License: MIT
 *
 * 1. Selecting a preset updates all light sliders instantly.
 * 2. Adjusting any slider automatically flips preset to "Custom".
 * 3. Pure 1-to-1 widget mapping: Never mutates node.widgets array so
 *    ComfyUI serialization/deserialization remains 100% mathematically exact.
 */

import { app } from "../../scripts/app.js";

const PRESET_MAP = {
    "Rembrandt Classic": {
        ambient_intensity: 0.05,
        key_enabled: true, key_azimuth: 45, key_elevation: 35, key_intensity: 1.0, key_kelvin: 5500, key_modifier: "Octabox (Soft)",
        fill_enabled: true, fill_azimuth: -30, fill_elevation: 15, fill_intensity: 0.35, fill_kelvin: 4800, fill_modifier: "Octabox (Soft)",
        rim_enabled: true, rim_azimuth: 160, rim_elevation: 45, rim_intensity: 1.1, rim_kelvin: 6500, rim_modifier: "Stripbox",
    },
    "High-Fashion Beauty": {
        ambient_intensity: 0.08,
        key_enabled: true, key_azimuth: 15, key_elevation: 45, key_intensity: 1.25, key_kelvin: 5600, key_modifier: "Beauty Dish (Silver)",
        fill_enabled: true, fill_azimuth: -15, fill_elevation: -10, fill_intensity: 0.40, fill_kelvin: 5500, fill_modifier: "Octabox (Soft)",
        rim_enabled: true, rim_azimuth: 140, rim_elevation: 40, rim_intensity: 0.90, rim_kelvin: 6500, rim_modifier: "Stripbox",
    },
    "Direct On-Camera Flash": {
        ambient_intensity: 0.02,
        key_enabled: true, key_azimuth: 0, key_elevation: 5, key_intensity: 1.40, key_kelvin: 5500, key_modifier: "Direct On-Camera Flash",
        fill_enabled: false, fill_azimuth: -30, fill_elevation: 0, fill_intensity: 0.0, fill_kelvin: 5500, fill_modifier: "Octabox (Soft)",
        rim_enabled: false, rim_azimuth: 180, rim_elevation: 30, rim_intensity: 0.0, rim_kelvin: 5500, rim_modifier: "Stripbox",
    },
    "Cinematic Film Noir": {
        ambient_intensity: 0.01,
        key_enabled: true, key_azimuth: 75, key_elevation: 20, key_intensity: 1.50, key_kelvin: 3200, key_modifier: "Fresnel Spotlight",
        fill_enabled: false, fill_azimuth: -45, fill_elevation: 0, fill_intensity: 0.05, fill_kelvin: 3200, fill_modifier: "Octabox (Soft)",
        rim_enabled: true, rim_azimuth: -150, rim_elevation: 35, rim_intensity: 1.40, rim_kelvin: 6500, rim_modifier: "Stripbox",
    },
    "Golden Hour Window": {
        ambient_intensity: 0.04,
        key_enabled: true, key_azimuth: -60, key_elevation: 15, key_intensity: 1.30, key_kelvin: 3000, key_modifier: "Octabox (Soft)",
        fill_enabled: true, fill_azimuth: 45, fill_elevation: 20, fill_intensity: 0.35, fill_kelvin: 6500, fill_modifier: "Octabox (Soft)",
        rim_enabled: true, rim_azimuth: -135, rim_elevation: 25, rim_intensity: 0.90, rim_kelvin: 3200, rim_modifier: "Stripbox",
    }
};

// Asynchronously fetch presets from Python backend
fetch("/studiolight/presets")
    .then((r) => r.json())
    .then((data) => {
        if (data && typeof data === "object") {
            Object.assign(PRESET_MAP, data);
        }
    })
    .catch(() => {});

function hookNodePresets(node) {
    if (!node || !node.widgets || node._sl_preset_hooked) return;
    node._sl_preset_hooked = true;

    const presetWidget = node.widgets.find((w) => w.name === "preset");
    if (!presetWidget) return;

    // 1. Hook Preset Dropdown Change
    const origPresetCb = presetWidget.callback;
    presetWidget.callback = function (val) {
        if (origPresetCb) origPresetCb.apply(this, arguments);
        const name = typeof val === "string" ? val : this.value;
        if (!name || name === "Custom" || !PRESET_MAP[name]) return;

        node._sl_applying = true;
        const p = PRESET_MAP[name];
        for (const [k, v] of Object.entries(p)) {
            const target = node.widgets.find((w) => w.name === k);
            if (target) {
                target.value = v;
                if (target.inputEl) target.inputEl.value = v;
                if (target.element) target.element.value = v;
            }
        }
        node._sl_applying = false;
        node.setDirtyCanvas(true, true);
    };

    // 2. Hook All Setting Widgets -> Auto-flip preset to "Custom" on tweak
    for (const w of node.widgets) {
        if (w.name === "preset") continue;
        const origCb = w.callback;
        w.callback = function () {
            if (origCb) origCb.apply(this, arguments);
            if (node._sl_applying) return;
            if (presetWidget.value !== "Custom") {
                presetWidget.value = "Custom";
                if (presetWidget.inputEl) presetWidget.inputEl.value = "Custom";
                if (presetWidget.element) presetWidget.element.value = "Custom";
                node.setDirtyCanvas(true, true);
            }
        };
    }

    // 3. Decorate Light Headers (Key, Fill, Rim section tag banners)
    decorateLightHeader(node, "key_enabled", "KEY LIGHT", "#FFA000", "rgba(255, 160, 0, 0.14)", "rgba(255, 160, 0, 0.45)", "#FFB300");
    decorateLightHeader(node, "fill_enabled", "FILL LIGHT", "#00B4D8", "rgba(0, 180, 216, 0.14)", "rgba(0, 180, 216, 0.45)", "#48CAE4");
    decorateLightHeader(node, "rim_enabled", "RIM / KICKER LIGHT", "#C77DFF", "rgba(199, 125, 255, 0.14)", "rgba(199, 125, 255, 0.45)", "#E0AAFF");

    const sz = node.computeSize();
    if (node.size && node.size[1] < sz[1]) {
        node.setSize([node.size[0], sz[1]]);
    }
}

function decorateLightHeader(node, widgetName, badgeTitle, badgeColor, bgColor, borderColor, textColor) {
    const w = node.widgets.find((x) => x.name === widgetName);
    if (!w || w._sl_header_decorated) return;
    w._sl_header_decorated = true;

    // Total widget height: 14px top separation margin + 24px header pill = 38px
    const TOP_MARGIN = 14;
    const PILL_H = 22;
    const TOTAL_H = TOP_MARGIN + PILL_H + 2;

    w.computeSize = function (width) {
        return [width, TOTAL_H];
    };

    w.draw = function (ctx, n, width, y, height) {
        ctx.save();
        const margin = 8;
        const pillX = margin;
        const pillW = width - margin * 2;
        // Align pill to lower section by placing top margin above it
        const pillY = y + TOP_MARGIN;
        const isEnabled = Boolean(w.value);

        // 1. Sleek badge container pill
        ctx.fillStyle = isEnabled ? bgColor : "rgba(255, 255, 255, 0.03)";
        ctx.strokeStyle = isEnabled ? borderColor : "rgba(255, 255, 255, 0.12)";
        ctx.lineWidth = 1.0;
        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(pillX, pillY, pillW, PILL_H, 4);
        } else {
            ctx.rect(pillX, pillY, pillW, PILL_H);
        }
        ctx.fill();
        ctx.stroke();

        // 2. LED Status Dot
        ctx.fillStyle = isEnabled ? badgeColor : "#555555";
        ctx.beginPath();
        ctx.arc(pillX + 12, pillY + PILL_H / 2, 3.5, 0, Math.PI * 2);
        ctx.fill();

        if (isEnabled) {
            ctx.fillStyle = badgeColor + "44";
            ctx.beginPath();
            ctx.arc(pillX + 12, pillY + PILL_H / 2, 6, 0, Math.PI * 2);
            ctx.fill();
        }

        // 3. Section Tag Title
        ctx.font = "bold 10px -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif";
        ctx.fillStyle = isEnabled ? textColor : "#777777";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(badgeTitle, pillX + 22, pillY + PILL_H / 2);

        // 4. Integrated ON / OFF Toggle Pill Button (Right side)
        const btnW = 38;
        const btnH = 15;
        const btnX = pillX + pillW - btnW - 4;
        const btnY = pillY + (PILL_H - btnH) / 2;

        ctx.beginPath();
        if (ctx.roundRect) {
            ctx.roundRect(btnX, btnY, btnW, btnH, 3);
        } else {
            ctx.rect(btnX, btnY, btnW, btnH);
        }

        if (isEnabled) {
            ctx.fillStyle = badgeColor;
            ctx.fill();
            ctx.fillStyle = "#111111";
            ctx.font = "bold 9px sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("ON", btnX + btnW / 2, btnY + btnH / 2);
        } else {
            ctx.fillStyle = "rgba(255, 255, 255, 0.08)";
            ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = "#888888";
            ctx.font = "bold 9px sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("OFF", btnX + btnW / 2, btnY + btnH / 2);
        }

        ctx.restore();
    };

    // Toggle on click
    w.mouse = function (event, pos, n) {
        if (event.type !== "pointerdown" && event.type !== "mousedown") return false;
        const localY = pos[1] - (this.last_y || 0);
        // Only trigger if clicked inside the pill (not in the top separation margin)
        if (localY >= TOP_MARGIN && localY <= TOP_MARGIN + PILL_H + 2) {
            w.value = !w.value;
            if (w.callback) {
                w.callback(w.value);
            }
            n.setDirtyCanvas(true, true);
            return true;
        }
        return false;
    };
}

function cleanObsoleteSlots(node) {
    if (!node) return;

    const widgetNames = (node.widgets || []).map((w) => w.name);
    const VALID_INPUTS = new Set(["normals_optional", "depth_optional", ...widgetNames]);
    const VALID_OUTPUTS = new Set(["relighting_map", "probe_preview", "lighting_prompt", "specular_pass", "telemetry", "shared_rig"]);
    const OBSOLETE_WIDGETS = new Set(["start_step", "end_step", "conditioning_strength", "weight_type"]);

    let changed = false;

    // 1. Remove obsolete inputs (backward loop)
    if (node.inputs) {
        for (let i = node.inputs.length - 1; i >= 0; i--) {
            const inp = node.inputs[i];
            if (!VALID_INPUTS.has(inp.name)) {
                if (typeof node.removeInput === "function") {
                    node.removeInput(i);
                } else {
                    node.inputs.splice(i, 1);
                }
                changed = true;
            }
        }
    }

    // 2. Remove obsolete outputs (backward loop)
    if (node.outputs) {
        for (let i = node.outputs.length - 1; i >= 0; i--) {
            const out = node.outputs[i];
            if (!VALID_OUTPUTS.has(out.name)) {
                if (typeof node.removeOutput === "function") {
                    node.removeOutput(i);
                } else {
                    node.outputs.splice(i, 1);
                }
                changed = true;
            }
        }
    }

    // 3. Remove obsolete widgets (backward loop)
    if (node.widgets) {
        for (let i = node.widgets.length - 1; i >= 0; i--) {
            const w = node.widgets[i];
            if (OBSOLETE_WIDGETS.has(w.name)) {
                node.widgets.splice(i, 1);
                changed = true;
            }
        }
    }

    if (changed) {
        const sz = node.computeSize();
        node.setSize([node.size[0], sz[1]]);
        node.setDirtyCanvas(true, true);
    }
}

function cleanCompositeNormalsInput(node) {
    const kind = node.comfyClass || node.type;
    if (!["StudioPhotometricComposite", "StudioPhotometricCompositeBeta"].includes(kind)) return;
    let changed = false;
    for (let i = (node.inputs?.length || 0) - 1; i >= 0; i--) {
        if (node.inputs[i].name === "normals") {
            // LiteGraph updates links and shifted target slots when an input is removed.
            node.removeInput(i);
            changed = true;
        }
    }
    if (changed) node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "ComfyUI.StudioLight.UI",

    async setup() {
        if (app.graph && app.graph._nodes) {
            for (const n of app.graph._nodes) {
                cleanCompositeNormalsInput(n);
                if (["StudioLightRig", "StudioLightConsole"].includes(n.comfyClass || n.type)) {
                    cleanObsoleteSlots(n);
                    hookNodePresets(n);
                }
            }
        }
    },

    nodeCreated(node) {
        cleanCompositeNormalsInput(node);
        if (["StudioLightRig", "StudioLightConsole"].includes(node.comfyClass || node.type)) {
            cleanObsoleteSlots(node);
            hookNodePresets(node);
        }
    },

    loadedGraphNode(node) {
        cleanCompositeNormalsInput(node);
        if (["StudioLightRig", "StudioLightConsole"].includes(node.comfyClass || node.type)) {
            cleanObsoleteSlots(node);
            hookNodePresets(node);
        }
    }
});
