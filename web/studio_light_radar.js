/**
 * ComfyUI-StudioLight — Interactive 2D Studio Radar & Live Preset Synchronizer
 * Author: Maciej Tajber (https://github.com/maciejtajber)
 * License: MIT
 *
 * Architecture: Clean onDrawForeground + prototype mouse event capture.
 * Zero custom widget mutation, zero closure-chaining in render loop.
 * 100% resilient across modern and legacy canvas engines.
 */

import { app } from "../../scripts/app.js";

// ─── Preset data ────────────────────────────────────────────────────────────
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

// Hydrate from backend dynamically (best-effort, fallback to static)
fetch("/studiolight/presets")
    .then(function (r) { return r.json(); })
    .then(function (data) {
        if (data && typeof data === "object") {
            Object.assign(PRESET_MAP, data);
        }
    })
    .catch(function () {});


// ─── Constants ──────────────────────────────────────────────────────────────
var RADAR_BOX_HEIGHT = 185;
var RADAR_PADDING    = 12;
var MIN_NODE_WIDTH   = 380;


// ─── Helpers ────────────────────────────────────────────────────────────────

function getWidget(node, name) {
    if (!node || !node.widgets) return undefined;
    for (var i = 0; i < node.widgets.length; i++) {
        if (node.widgets[i].name === name) return node.widgets[i];
    }
    return undefined;
}

function getWidgetValue(node, name, fallback) {
    var w = getWidget(node, name);
    return (w !== undefined && w.value !== undefined) ? w.value : fallback;
}

function setWidgetValue(node, name, val) {
    var w = getWidget(node, name);
    if (!w) return;
    w.value = val;
    if (w.inputEl) w.inputEl.value = val;
    if (w.element) w.element.value = val;
    if (w.callback) {
        try { w.callback.call(w, val); } catch (e) { /* swallow */ }
    }
}

function safeRoundRect(ctx, x, y, w, h, r) {
    if (typeof ctx.roundRect === "function") {
        ctx.roundRect(x, y, w, h, r);
    } else {
        ctx.rect(x, y, w, h);
    }
}


// ─── Preset synchronization logic ───────────────────────────────────────────

function applyPreset(node, presetName) {
    if (!presetName || presetName === "Custom" || !PRESET_MAP[presetName]) return;
    node._sl_applying = true;
    var p = PRESET_MAP[presetName];
    for (var k in p) {
        if (p.hasOwnProperty(k)) {
            setWidgetValue(node, k, p[k]);
        }
    }
    node._sl_applying = false;
    node.setDirtyCanvas(true, true);
    if (app.canvas) app.canvas.setDirty(true, true);
}

function flipToCustom(node) {
    if (!node || node._sl_applying) return;
    var pw = getWidget(node, "preset");
    if (pw && pw.value !== "Custom") {
        pw.value = "Custom";
        if (pw.inputEl) pw.inputEl.value = "Custom";
        if (pw.element) pw.element.value = "Custom";
        node.setDirtyCanvas(true, true);
        if (app.canvas) app.canvas.setDirty(true, true);
    }
}

function hookPresetWidget(node) {
    if (!node) return;
    var pw = getWidget(node, "preset");
    if (pw && !pw._sl_hooked) {
        pw._sl_hooked = true;
        var origCb = pw.callback;
        pw.callback = function (val) {
            if (origCb) origCb.apply(this, arguments);
            applyPreset(node, typeof val === "string" ? val : this.value);
        };
    }
}


// ─── Compute radar geometry relative to node ────────────────────────────────

function radarMetrics(node) {
    var w = (node.size && node.size[0]) ? node.size[0] : MIN_NODE_WIDTH;
    var h = (node.size && node.size[1]) ? node.size[1] : 750;
    var boxW = w - RADAR_PADDING * 2;
    var boxH = RADAR_BOX_HEIGHT;
    var boxX = RADAR_PADDING;
    var boxY = h - boxH - 8;
    var cx = w / 2;
    var cy = boxY + boxH / 2 + 4;
    var radius = Math.min(boxW, boxH) / 2 - 20;
    if (radius < 24) radius = 24;
    return { w: w, h: h, boxX: boxX, boxY: boxY, boxW: boxW, boxH: boxH, cx: cx, cy: cy, radius: radius };
}


// ─── Drawing ────────────────────────────────────────────────────────────────

function drawRadar(node, ctx) {
    var m = radarMetrics(node);

    // 1. Pod container background
    ctx.fillStyle = "#111116";
    ctx.strokeStyle = "#383846";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    safeRoundRect(ctx, m.boxX, m.boxY, m.boxW, m.boxH, 5);
    ctx.fill();
    ctx.stroke();

    // 2. Pod Title Header
    ctx.fillStyle = "#9a9aa8";
    ctx.font = "bold 9px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("3-POINT STUDIO RADAR (DRAG LIGHTS TO AIM)", m.cx, m.boxY + 14);

    // 3. Circular radar concentric rings
    ctx.strokeStyle = "#252530";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(m.cx, m.cy, m.radius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(m.cx, m.cy, m.radius * 0.5, 0, Math.PI * 2);
    ctx.stroke();

    // 4. Crosshairs
    ctx.strokeStyle = "#202028";
    ctx.beginPath();
    ctx.moveTo(m.cx - m.radius, m.cy);
    ctx.lineTo(m.cx + m.radius, m.cy);
    ctx.moveTo(m.cx, m.cy - m.radius);
    ctx.lineTo(m.cx, m.cy + m.radius);
    ctx.stroke();

    // 5. Diagonal guidelines (+/- 45°, +/- 135°)
    ctx.strokeStyle = "#1a1a22";
    var diagAngles = [45, 135, -45, -135];
    for (var i = 0; i < diagAngles.length; i++) {
        var rad = diagAngles[i] * (Math.PI / 180);
        ctx.beginPath();
        ctx.moveTo(m.cx, m.cy);
        ctx.lineTo(m.cx + m.radius * Math.sin(rad), m.cy + m.radius * Math.cos(rad));
        ctx.stroke();
    }

    // 6. Camera marker at bottom (0°) - Operator perspective behind camera
    ctx.fillStyle = "#3e3e50";
    ctx.strokeStyle = "#5a5a70";
    ctx.lineWidth = 1;
    ctx.beginPath();
    safeRoundRect(ctx, m.cx - 40, m.cy + m.radius + 3, 80, 13, 3);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 8px monospace";
    ctx.textAlign = "center";
    ctx.fillText("▼ CAMERA (0°)", m.cx, m.cy + m.radius + 12);

    // 7. Center subject point
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(m.cx, m.cy, 4, 0, Math.PI * 2);
    ctx.fill();

    // 8. Glowing neon light handles
    var lights = [
        { azName: "key_azimuth", enName: "key_enabled", elName: "key_elevation", color: "#ffaa00", label: "KEY", defAz: 45, defEl: 35 },
        { azName: "fill_azimuth", enName: "fill_enabled", elName: "fill_elevation", color: "#00d4ff", label: "FILL", defAz: -30, defEl: 15 },
        { azName: "rim_azimuth", enName: "rim_enabled", elName: "rim_elevation", color: "#ff2a85", label: "RIM", defAz: 160, defEl: 45 },
    ];

    for (var li = 0; li < lights.length; li++) {
        var info = lights[li];
        var enabled = getWidgetValue(node, info.enName, true);
        if (!enabled) continue;

        var az = getWidgetValue(node, info.azName, info.defAz);
        var rad = az * (Math.PI / 180);
        var lx = m.cx + m.radius * Math.sin(rad);
        var ly = m.cy + m.radius * Math.cos(rad);

        // Ray line from center to light
        ctx.strokeStyle = info.color;
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        ctx.moveTo(m.cx, m.cy);
        ctx.lineTo(lx, ly);
        ctx.stroke();

        // Glowing outer handle ring
        ctx.fillStyle = info.color;
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
        ctx.fillStyle = info.color;
        ctx.font = "bold 9px sans-serif";
        var isRight = Math.sin(rad) >= 0;
        ctx.textAlign = isRight ? "left" : "right";
        ctx.fillText(info.label + " " + Math.round(az) + "°", lx + (isRight ? 11 : -11), ly + 3);
    }

    // Elevation readout in bottom corner
    ctx.fillStyle = "#707084";
    ctx.font = "8px monospace";
    ctx.textAlign = "left";
    var kEl = Math.round(getWidgetValue(node, "key_elevation", 35));
    var fEl = Math.round(getWidgetValue(node, "fill_elevation", 15));
    var rEl = Math.round(getWidgetValue(node, "rim_elevation", 45));
    ctx.fillText("EL  K:" + kEl + "°  F:" + fEl + "°  R:" + rEl + "°", m.boxX + 8, m.boxY + m.boxH - 5);
}


// ─── Mouse interaction ──────────────────────────────────────────────────────

function hitTestRadar(node, localX, localY) {
    var m = radarMetrics(node);

    // Must be inside the radar box
    if (localY < m.boxY || localY > m.boxY + m.boxH) return null;
    if (localX < m.boxX || localX > m.boxX + m.boxW) return null;

    var lights = [
        { azName: "key_azimuth", enName: "key_enabled", defAz: 45 },
        { azName: "fill_azimuth", enName: "fill_enabled", defAz: -30 },
        { azName: "rim_azimuth", enName: "rim_enabled", defAz: 160 },
    ];

    var bestDist = 999;
    var bestWidget = null;

    for (var i = 0; i < lights.length; i++) {
        var info = lights[i];
        if (!getWidgetValue(node, info.enName, true)) continue;
        var az = getWidgetValue(node, info.azName, info.defAz);
        var rad = az * (Math.PI / 180);
        var lx = m.cx + m.radius * Math.sin(rad);
        var ly = m.cy + m.radius * Math.cos(rad);
        var dist = Math.hypot(localX - lx, localY - ly);
        if (dist < 24 && dist < bestDist) {
            bestDist = dist;
            bestWidget = info.azName;
        }
    }

    // If no handle was directly clicked but click is inside the circular arena, default to Key
    if (!bestWidget) {
        var distFromCenter = Math.hypot(localX - m.cx, localY - m.cy);
        if (distFromCenter <= m.radius + 15) {
            bestWidget = "key_azimuth";
        }
    }

    return bestWidget;
}

function aimLight(node, localX, localY, widgetName) {
    var m = radarMetrics(node);
    var dx = localX - m.cx;
    var dy = localY - m.cy;
    // Camera at bottom (0°): dy > 0 is bottom (0°), dx > 0 is right (+90°), dy < 0 is top (+/-180°)
    var deg = Math.round(Math.atan2(dx, dy) * (180 / Math.PI));
    setWidgetValue(node, widgetName, deg);
    flipToCustom(node);
    node.setDirtyCanvas(true, true);
    if (app.canvas) app.canvas.setDirty(true, true);
}


// ─── Extension registration ─────────────────────────────────────────────────

app.registerExtension({
    name: "ComfyUI.StudioLight.Radar",

    beforeRegisterNodeDef: function (nodeType, nodeData) {
        if (nodeData.name !== "StudioLightRig" && nodeType.comfyClass !== "StudioLightRig") return;

        // ── Allocate layout space on prototype computeSize (ONCE, zero closure nesting) ──
        var origComputeSize = nodeType.prototype.computeSize;
        nodeType.prototype.computeSize = function (out) {
            var s = origComputeSize ? origComputeSize.apply(this, arguments) : [MIN_NODE_WIDTH, 700];
            var res = out || s;
            res[0] = Math.max(res[0] || 0, MIN_NODE_WIDTH);
            res[1] = Math.max(res[1] || 0, (s[1] || 700) + RADAR_BOX_HEIGHT + 24);
            return res;
        };

        // ── Draw the radar below all widgets in foreground pass ──
        var origDrawFg = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            if (origDrawFg) origDrawFg.call(this, ctx);
            if (this.flags && this.flags.collapsed) return;

            // Ensure node size accounts for the radar box
            if (!this._sl_sized) {
                this._sl_sized = true;
                var baseSize = (typeof this.computeSize === "function") ? this.computeSize() : [MIN_NODE_WIDTH, 750];
                if (!this.size) this.size = [MIN_NODE_WIDTH, baseSize[1]];
                if (this.size[0] < MIN_NODE_WIDTH) this.size[0] = MIN_NODE_WIDTH;
                if (this.size[1] < baseSize[1]) this.size[1] = baseSize[1];
            }

            try {
                ctx.save();
                drawRadar(this, ctx);
                ctx.restore();
            } catch (e) {
                try { ctx.restore(); } catch (err) {}
                console.warn("[StudioLight Radar] draw error:", e);
            }
        };

        // ── Mouse down: start interactive aiming drag ──
        var origMouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function (e, localPos, graphCanvas) {
            var hit = hitTestRadar(this, localPos[0], localPos[1]);
            if (hit) {
                this._sl_drag = hit;
                aimLight(this, localPos[0], localPos[1], hit);
                this.captureInput = true;
                if (graphCanvas) graphCanvas.node_capturing_input = this;
                return true;
            }
            return origMouseDown ? origMouseDown.call(this, e, localPos, graphCanvas) : false;
        };

        // ── Mouse move: drag to rotate light vector in real-time ──
        var origMouseMove = nodeType.prototype.onMouseMove;
        nodeType.prototype.onMouseMove = function (e, localPos, graphCanvas) {
            if (this._sl_drag) {
                aimLight(this, localPos[0], localPos[1], this._sl_drag);
                return true;
            }
            return origMouseMove ? origMouseMove.call(this, e, localPos, graphCanvas) : false;
        };

        // ── Mouse up: release drag cleanly ──
        var origMouseUp = nodeType.prototype.onMouseUp;
        nodeType.prototype.onMouseUp = function (e, localPos, graphCanvas) {
            if (this._sl_drag) {
                this._sl_drag = null;
                this.captureInput = false;
                if (graphCanvas && graphCanvas.node_capturing_input === this) {
                    graphCanvas.node_capturing_input = null;
                }
                return true;
            }
            return origMouseUp ? origMouseUp.call(this, e, localPos, graphCanvas) : false;
        };

        // ── Widget changed: immediately auto-flip preset dropdown to "Custom" ──
        var origWidgetChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function (name, value, old_value, widget) {
            if (origWidgetChanged) origWidgetChanged.call(this, name, value, old_value, widget);
            if (name !== "preset" && name !== "framing") {
                flipToCustom(this);
            }
        };
    },

    nodeCreated: function (node) {
        if (node.comfyClass === "StudioLightRig" || node.type === "StudioLightRig") {
            hookPresetWidget(node);
        }
    },

    loadedGraphNode: function (node) {
        if (node.comfyClass === "StudioLightRig" || node.type === "StudioLightRig") {
            hookPresetWidget(node);
        }
    },
});
