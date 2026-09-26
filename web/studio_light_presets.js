/**
 * ComfyUI-StudioLight — Live Preset Synchronizer & Visual Section Dividers
 * Author: Maciej Tajber (https://github.com/maciejtajber)
 * License: MIT
 *
 * 1. Selecting a preset updates all light sliders instantly.
 * 2. Adjusting any slider automatically flips preset to "Custom".
 * 3. Injects sleek, non-serialized section dividers between Key, Fill, Rim, and Scheduling.
 * 4. Zero canvas drawing hooks on the node, zero size recursion, 100% crash-proof.
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

// ─── Visual Section Divider Constructor ──────────────────────────────────────
function createDividerWidget(id, label, color) {
    return {
        name: "_sl_div_" + id,
        type: "custom",
        value: "",
        serialize: false,
        options: { serialize: false, canvasOnly: true },
        computeSize(width) {
            return [width || 360, 24];
        },
        draw(ctx, targetNode, widget_width, y, widget_height) {
            const w = widget_width || (targetNode.size ? targetNode.size[0] : 360);
            const midY = y + 12;

            ctx.save();

            // 1. Subtle horizontal divider rule
            ctx.strokeStyle = "#343442";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(10, midY);
            ctx.lineTo(w - 10, midY);
            ctx.stroke();

            // 2. Section pill badge
            ctx.font = "bold 9px sans-serif";
            const textWidth = ctx.measureText(label).width;
            const badgeWidth = textWidth + 22;
            const badgeX = 16;
            const badgeY = midY - 8;

            // Pill container background
            ctx.fillStyle = "#16161e";
            ctx.strokeStyle = color;
            ctx.lineWidth = 1;
            ctx.beginPath();
            if (typeof ctx.roundRect === "function") {
                ctx.roundRect(badgeX, badgeY, badgeWidth, 16, 3);
            } else {
                ctx.rect(badgeX, badgeY, badgeWidth, 16);
            }
            ctx.fill();
            ctx.stroke();

            // Glowing accent dot
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(badgeX + 8, midY, 2.8, 0, Math.PI * 2);
            ctx.fill();

            // Label text
            ctx.fillStyle = "#e0e0ea";
            ctx.textAlign = "left";
            ctx.fillText(label, badgeX + 15, midY + 3);

            ctx.restore();
        },
        mouse() {
            return false;
        }
    };
}

function insertSectionDividers(node) {
    if (!node || !node.widgets || node._sl_dividers_added) return;

    // Check if dividers already exist
    if (node.widgets.some((w) => w.name && w.name.startsWith("_sl_div_"))) {
        node._sl_dividers_added = true;
        return;
    }

    node._sl_dividers_added = true;

    const sections = [
        { target: "key_enabled", id: "key", label: "KEY LIGHT", color: "#ffaa00" },
        { target: "fill_enabled", id: "fill", label: "FILL LIGHT", color: "#00d4ff" },
        { target: "rim_enabled", id: "rim", label: "RIM / KICKER LIGHT", color: "#ff2a85" },
        { target: "start_step", id: "sched", label: "SCHEDULING & CONDITIONING", color: "#8e8ea0" },
    ];

    for (const sec of sections) {
        const idx = node.widgets.findIndex((w) => w.name === sec.target);
        if (idx !== -1) {
            const div = createDividerWidget(sec.id, sec.label, sec.color);
            node.widgets.splice(idx, 0, div);
        }
    }

    // Expand node height once cleanly
    if (node.size) {
        node.size[1] = Math.max(node.size[1] || 0, 700) + 24 * sections.length;
    }
    node.setDirtyCanvas(true, true);
}

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
        if (w.name.startsWith("_sl_div_") || w.name === "preset" || w.name === "framing") continue;
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
            insertSectionDividers(node);
            hookNodePresets(node);
        }
    },

    loadedGraphNode(node) {
        if (node.comfyClass === "StudioLightRig" || node.type === "StudioLightRig") {
            insertSectionDividers(node);
            hookNodePresets(node);
        }
    }
});
