/**
 * ComfyUI-StudioLight — Interactive 2D Studio Radar & Live Preset Synchronizer
 * Author: Maciej Tajber (https://github.com/maciejtajber)
 * License: MIT
 */

import { app } from "../../scripts/app.js";

// Static fallback in case backend is offline; dynamically hydrated via /studiolight/presets
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

// Asynchronously fetch presets from Python backend for 100% Single Source of Truth
fetch("/studiolight/presets")
    .then((r) => r.json())
    .then((data) => {
        if (data && typeof data === "object") {
            Object.assign(PRESET_MAP, data);
        }
    })
    .catch(() => {});

const RADAR_HEIGHT = 185;

function getWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function applyPresetValues(node, currentVal) {
    if (!currentVal || currentVal === "Custom" || !PRESET_MAP[currentVal]) return;
    node._isApplyingPreset = true;
    const p = PRESET_MAP[currentVal];
    for (const [k, v] of Object.entries(p)) {
        const targetWidget = getWidget(node, k);
        if (targetWidget) {
            targetWidget.value = v;
            if (targetWidget.inputEl) targetWidget.inputEl.value = v;
            if (targetWidget.element) targetWidget.element.value = v;
            if (targetWidget.callback) {
                try {
                    targetWidget.callback.call(targetWidget, v);
                } catch (e) {}
            }
        }
    }
    node._isApplyingPreset = false;
    node.setDirtyCanvas(true, true);
    app.canvas?.setDirty(true, true);
}

function setPresetToCustom(node) {
    if (!node || node._isApplyingPreset) return;
    const pw = getWidget(node, "preset");
    if (pw && pw.value !== "Custom") {
        pw.value = "Custom";
        if (pw.inputEl) pw.inputEl.value = "Custom";
        if (pw.element) pw.element.value = "Custom";
        node.setDirtyCanvas(true, true);
        app.canvas?.setDirty(true, true);
    }
}

function updateLightAngle(node, dx, dy, widgetName) {
    const w = getWidget(node, widgetName);
    if (!w) return;
    // Camera at bottom (0°): dy > 0 is bottom (0°), dx > 0 is right (+90°), dy < 0 is top (+/-180°)
    let deg = Math.atan2(dx, dy) * (180 / Math.PI);
    deg = Math.round(deg);
    w.value = deg;
    if (w.inputEl) w.inputEl.value = deg;
    if (w.element) w.element.value = deg;
    if (w.callback) {
        try {
            w.callback.call(w, w.value);
        } catch (e) {}
    }
    setPresetToCustom(node);
    node.setDirtyCanvas(true, true);
    app.canvas?.setDirty(true, true);
}

function drawRadar(node, ctx, yOffset, customHeight, customWidth) {
    if (node.flags?.collapsed) return;

    const width = customWidth || node.size[0] || 380;
    const radarY = typeof yOffset === "number" ? yOffset + 4 : (node.size[1] - RADAR_HEIGHT - 10);
    const boxHeight = customHeight ? customHeight - 8 : RADAR_HEIGHT - 8;
    const cx = width / 2;
    const cy = radarY + boxHeight / 2 + 4;
    const radius = Math.min(width, boxHeight) / 2 - 20;

    // Cache metrics on node for mouse picking
    node._radarMetrics = { cx, cy, radius, radarY, boxHeight, width };

    ctx.save();

    // 1. Pod container background
    ctx.fillStyle = "#111116";
    ctx.strokeStyle = "#383846";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.rect(12, radarY, width - 24, boxHeight);
    ctx.fill();
    ctx.stroke();

    // 2. Pod Title Header
    ctx.fillStyle = "#9a9aa8";
    ctx.font = "bold 9px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("3-POINT STUDIO RADAR (DRAG LIGHTS TO AIM)", cx, radarY + 14);

    // 3. Circular radar concentric rings
    ctx.strokeStyle = "#252530";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(cx, cy, radius * 0.5, 0, Math.PI * 2);
    ctx.stroke();

    // 4. Crosshairs
    ctx.strokeStyle = "#202028";
    ctx.beginPath();
    ctx.moveTo(cx - radius, cy);
    ctx.lineTo(cx + radius, cy);
    ctx.moveTo(cx, cy - radius);
    ctx.lineTo(cx, cy + radius);
    ctx.stroke();

    // 5. Diagonal guidelines (+/- 45°, +/- 135°)
    ctx.strokeStyle = "#1a1a22";
    for (const angle of [45, 135, -45, -135]) {
        const rad = angle * (Math.PI / 180);
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + radius * Math.sin(rad), cy + radius * Math.cos(rad));
        ctx.stroke();
    }

    // 6. Camera marker at bottom (0°) - Operator perspective behind camera
    ctx.fillStyle = "#3e3e50";
    ctx.strokeStyle = "#5a5a70";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(cx - 40, cy + radius + 2, 80, 14);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 8px monospace";
    ctx.textAlign = "center";
    ctx.fillText("▼ CAMERA (0°)", cx, cy + radius + 12);

    // 7. Center subject point
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.fill();

    // 8. Helper to render ray, handle, and degree badge
    function drawLightHandle(azimuth, color, label, enabled) {
        if (!enabled) return;

        // 0° is bottom (camera), +45° is camera-right (screen right), +/-180° is top (backlight)
        const rad = azimuth * (Math.PI / 180);
        const lx = cx + radius * Math.sin(rad);
        const ly = cy + radius * Math.cos(rad);

        // Ray line from center to light
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(lx, ly);
        ctx.stroke();

        // Glowing outer handle ring
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(lx, ly, 7.5, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.6;
        ctx.stroke();

        // Inner white dot
        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(lx, ly, 2.5, 0, Math.PI * 2);
        ctx.fill();

        // Text label badge
        ctx.fillStyle = color;
        ctx.font = "bold 9px sans-serif";
        const isRight = Math.sin(rad) >= 0;
        ctx.textAlign = isRight ? "left" : "right";
        ctx.fillText(`${label} ${Math.round(azimuth)}°`, lx + (isRight ? 11 : -11), ly + 3);
    }

    const kAz = getWidget(node, "key_azimuth")?.value ?? 45;
    const kEn = getWidget(node, "key_enabled")?.value ?? true;

    const fAz = getWidget(node, "fill_azimuth")?.value ?? -30;
    const fEn = getWidget(node, "fill_enabled")?.value ?? true;

    const rAz = getWidget(node, "rim_azimuth")?.value ?? 160;
    const rEn = getWidget(node, "rim_enabled")?.value ?? true;

    // Draw active lights with distinct high-contrast neon palettes
    drawLightHandle(kAz, "#ffaa00", "KEY", kEn);
    drawLightHandle(fAz, "#00d4ff", "FILL", fEn);
    drawLightHandle(rAz, "#ff2a85", "RIM", rEn);

    // Elevation readout in bottom corner
    const kEl = getWidget(node, "key_elevation")?.value ?? 35;
    const fEl = getWidget(node, "fill_elevation")?.value ?? 15;
    const rEl = getWidget(node, "rim_elevation")?.value ?? 45;
    ctx.fillStyle = "#707084";
    ctx.font = "8px monospace";
    ctx.textAlign = "left";
    ctx.fillText(`EL: K:${Math.round(kEl)}° F:${Math.round(fEl)}° R:${Math.round(rEl)}°`, 18, radarY + boxHeight - 5);

    ctx.restore();
}

function handleRadarMouse(event, localPos, node) {
    if (node.flags?.collapsed || !node._radarMetrics) return false;

    const { cx, cy, radius, radarY, boxHeight } = node._radarMetrics;
    const mx = localPos[0];
    const my = localPos[1];
    const dist = Math.hypot(mx - cx, my - cy);

    const type = event.type || "";

    if (type === "pointerdown" || type === "mousedown") {
        if (dist <= radius + 22 && my >= radarY && my <= radarY + boxHeight) {
            const lights = [
                { id: "key", widget: "key_azimuth" },
                { id: "fill", widget: "fill_azimuth" },
                { id: "rim", widget: "rim_azimuth" },
            ];

            for (const l of lights) {
                const w = getWidget(node, l.widget);
                if (!w) continue;
                const az = (w.value) * (Math.PI / 180);
                const lx = cx + radius * Math.sin(az);
                const ly = cy + radius * Math.cos(az);
                if (Math.hypot(mx - lx, my - ly) <= 22) {
                    node._draggingLight = l.widget;
                    updateLightAngle(node, mx - cx, my - cy, node._draggingLight);
                    return true;
                }
            }

            // Clicked inside radar ring -> aim closest or default Key
            node._draggingLight = "key_azimuth";
            updateLightAngle(node, mx - cx, my - cy, node._draggingLight);
            return true;
        }
    } else if (type === "pointermove" || type === "mousemove") {
        if (node._draggingLight) {
            updateLightAngle(node, mx - cx, my - cy, node._draggingLight);
            return true;
        }
    } else if (type === "pointerup" || type === "mouseup") {
        if (node._draggingLight) {
            node._draggingLight = null;
            return true;
        }
    }

    return false;
}

function setupRadarWidget(node) {
    if (!node || node._studio_widget_installed) return;
    node._studio_widget_installed = true;

    // Hook preset widget callback
    const presetWidget = getWidget(node, "preset");
    if (presetWidget && !presetWidget._studio_hooked) {
        presetWidget._studio_hooked = true;
        const origPresetCb = presetWidget.callback;
        presetWidget.callback = function (val) {
            if (origPresetCb) origPresetCb.apply(this, arguments);
            const currentVal = typeof val === "string" ? val : this.value;
            applyPresetValues(node, currentVal);
        };
    }

    // Add dedicated custom widget to node.widgets if not already present
    if (!node.widgets?.some((w) => w.name === "studio_radar")) {
        const radarWidget = {
            name: "studio_radar",
            type: "custom",
            value: "",
            options: { serialize: false },
            draw(ctx, targetNode, widget_width, y, widget_height) {
                drawRadar(targetNode, ctx, y, widget_height, widget_width);
            },
            mouse(event, pos, targetNode) {
                return handleRadarMouse(event, pos, targetNode);
            },
            computeSize(width) {
                return [width || 380, RADAR_HEIGHT];
            }
        };

        if (typeof node.addCustomWidget === "function") {
            node.addCustomWidget(radarWidget);
        } else {
            node.widgets = node.widgets || [];
            node.widgets.push(radarWidget);
        }
    }

    // Ensure node computeSize accounts for radar widget
    const origComputeSize = node.computeSize?.bind(node);
    node.computeSize = function () {
        const s = origComputeSize ? origComputeSize.apply(this, arguments) : [380, 880];
        s[0] = Math.max(s[0] || 0, 380);
        s[1] = Math.max(s[1] || 0, 880);
        return s;
    };

    // Ensure node size fits
    if (node.size) {
        node.size[0] = Math.max(node.size[0], 380);
        node.size[1] = Math.max(node.size[1], 880);
    }
}

app.registerExtension({
    name: "ComfyUI.StudioLight.Radar",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "StudioLightRig" && nodeType.comfyClass !== "StudioLightRig") {
            return;
        }

        // Prototype configure hook
        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            if (origConfigure) origConfigure.apply(this, arguments);
            setupRadarWidget(this);
        };

        // Prototype onDrawForeground fallback
        const origDrawFg = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            if (origDrawFg) origDrawFg.apply(this, arguments);
            setupRadarWidget(this);
            // If the custom widget didn't draw, draw directly as fallback
            if (!this.widgets?.some((w) => w.name === "studio_radar")) {
                drawRadar(this, ctx);
            }
        };

        // Prototype mouse handlers as robust fallback
        const origMouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function (e, localPos) {
            if (handleRadarMouse(e, localPos, this)) return true;
            return origMouseDown ? origMouseDown.apply(this, arguments) : false;
        };

        const origMouseMove = nodeType.prototype.onMouseMove;
        nodeType.prototype.onMouseMove = function (e, localPos) {
            if (handleRadarMouse(e, localPos, this)) return true;
            return origMouseMove ? origMouseMove.apply(this, arguments) : false;
        };

        const origMouseUp = nodeType.prototype.onMouseUp;
        nodeType.prototype.onMouseUp = function (e, localPos) {
            if (handleRadarMouse(e, localPos, this)) return true;
            return origMouseUp ? origMouseUp.apply(this, arguments) : false;
        };

        // onWidgetChanged: If ANY widget is adjusted, immediately flip preset to "Custom"
        const origWidgetChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function (name, value, old_value, widget) {
            if (origWidgetChanged) origWidgetChanged.apply(this, arguments);
            if (name !== "preset" && name !== "framing" && name !== "studio_radar") {
                setPresetToCustom(this);
            }
        };
    },

    nodeCreated(node) {
        if (node.type === "StudioLightRig" || node.comfyClass === "StudioLightRig") {
            setupRadarWidget(node);
        }
    },

    loadedGraphNode(node) {
        if (node.type === "StudioLightRig" || node.comfyClass === "StudioLightRig") {
            setupRadarWidget(node);
        }
    }
});
