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
        if (w.name === "preset" || w.name === "framing") continue;
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
}

app.registerExtension({
    name: "ComfyUI.StudioLight.UI",

    nodeCreated(node) {
        if (node.comfyClass === "StudioLightRig" || node.type === "StudioLightRig") {
            hookNodePresets(node);
        }
    },

    loadedGraphNode(node) {
        if (node.comfyClass === "StudioLightRig" || node.type === "StudioLightRig") {
            hookNodePresets(node);
        }
    }
});
