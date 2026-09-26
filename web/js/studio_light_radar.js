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

app.registerExtension({
    name: "ComfyUI.StudioLight.Radar",

    beforeRegisterNodeDef(nodeType, nodeData, appRef) {
        if (nodeData.name === "StudioLightRig") {
            const origOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
                app.extensions?.find(e => e.name === "ComfyUI.StudioLight.Radar")?.setupStudioLightNode(this);
                return r;
            };

            const origOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
                app.extensions?.find(e => e.name === "ComfyUI.StudioLight.Radar")?.setupStudioLightNode(this);
                return r;
            };
        }
    },

    async nodeCreated(node) {
        this.setupStudioLightNode(node);
    },
    async loadedGraphNode(node) {
        this.setupStudioLightNode(node);
    },

    setupStudioLightNode(node) {
        const nodeType = node.type || node.constructor?.type || node.comfyClass;
        if (nodeType !== "StudioLightRig") {
            return;
        }

        let isApplyingPreset = false;

        function getWidget(name) {
            return node.widgets?.find((w) => w.name === name);
        }

        function applyPresetValues(currentVal) {
            if (!currentVal || currentVal === "Custom" || !PRESET_MAP[currentVal]) return;
            isApplyingPreset = true;
            const p = PRESET_MAP[currentVal];
            for (const [k, v] of Object.entries(p)) {
                const targetWidget = getWidget(k);
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
            isApplyingPreset = false;
            node.setDirtyCanvas(true, true);
            app.canvas?.setDirty(true, true);
        }

        // 1. Hook Preset Dropdown: When selected, update all sliders visually
        const presetWidget = getWidget("preset");
        if (presetWidget && !presetWidget._studio_hooked) {
            presetWidget._studio_hooked = true;
            const originalPresetCallback = presetWidget.callback;
            presetWidget.callback = function (val) {
                if (originalPresetCallback) originalPresetCallback.apply(this, arguments);
                const currentVal = typeof val === "string" ? val : this.value;
                applyPresetValues(currentVal);
            };
        }

        // 2. If user adjusts any light slider manually, switch preset dropdown to "Custom"
        const origOnWidgetChanged = node.onWidgetChanged;
        node.onWidgetChanged = function (name, value, old_value, widget) {
            if (origOnWidgetChanged) origOnWidgetChanged.apply(this, arguments);
            if (!isApplyingPreset && name !== "preset" && name !== "framing" && presetWidget && presetWidget.value !== "Custom") {
                presetWidget.value = "Custom";
                if (presetWidget.inputEl) presetWidget.inputEl.value = "Custom";
                node.setDirtyCanvas(true, true);
                app.canvas?.setDirty(true, true);
            }
        };

        const monitoredSliders = [
            "ambient_intensity",
            "key_enabled", "key_azimuth", "key_elevation", "key_intensity", "key_kelvin", "key_modifier",
            "fill_enabled", "fill_azimuth", "fill_elevation", "fill_intensity", "fill_kelvin", "fill_modifier",
            "rim_enabled", "rim_azimuth", "rim_elevation", "rim_intensity", "rim_kelvin", "rim_modifier"
        ];

        for (const name of monitoredSliders) {
            const w = getWidget(name);
            if (w && !w._studio_hooked) {
                w._studio_hooked = true;
                const originalCb = w.callback;
                w.callback = function (val) {
                    if (originalCb) originalCb.apply(this, arguments);
                    if (!isApplyingPreset && presetWidget && presetWidget.value !== "Custom") {
                        presetWidget.value = "Custom";
                        if (presetWidget.inputEl) presetWidget.inputEl.value = "Custom";
                    }
                    node.setDirtyCanvas(true, true);
                    app.canvas?.setDirty(true, true);
                };
            }
        }

        // 3. Interactive Visual Studio Radar (Rendered in dedicated bottom console area)
        const RADAR_HEIGHT = 160;
        let draggingLight = null;

        // Ensure node height accounts for the bottom radar console
        const originalComputeSize = node.computeSize?.bind(node);
        node.computeSize = function () {
            const s = originalComputeSize ? originalComputeSize() : [380, 720];
            s[0] = Math.max(s[0], 380);
            s[1] = Math.max(s[1], 880);
            return s;
        };
        node.size[0] = Math.max(node.size[0], 380);
        node.size[1] = Math.max(node.size[1], 880);

        // Mouse click and drag to aim lights on the radar
        const originalOnMouseDown = node.onMouseDown?.bind(node);
        node.onMouseDown = function (e, localPos) {
            if (node.flags?.collapsed) return originalOnMouseDown ? originalOnMouseDown(e, localPos) : false;

            const width = node.size[0];
            const radarY = node.size[1] - RADAR_HEIGHT - 12;
            const cx = width / 2;
            const cy = radarY + RADAR_HEIGHT / 2;
            const radius = Math.min(width, RADAR_HEIGHT) / 2 - 20;

            const mx = localPos[0];
            const my = localPos[1];
            const dist = Math.hypot(mx - cx, my - cy);

            if (dist <= radius + 15 && my >= radarY && my <= radarY + RADAR_HEIGHT) {
                // Determine which light head was clicked
                const lights = [
                    { id: "key", widget: "key_azimuth" },
                    { id: "fill", widget: "fill_azimuth" },
                    { id: "rim", widget: "rim_azimuth" },
                ];

                for (const l of lights) {
                    const w = getWidget(l.widget);
                    if (!w) continue;
                    // Camera at bottom (0°): screen x = cx + r*sin(az), screen y = cy + r*cos(az)
                    const az = (w.value) * (Math.PI / 180);
                    const lx = cx + radius * Math.sin(az);
                    const ly = cy + radius * Math.cos(az);
                    if (Math.hypot(mx - lx, my - ly) <= 18) {
                        draggingLight = l.widget;
                        return true;
                    }
                }

                // If clicked inside radar circle, target Key by default
                draggingLight = "key_azimuth";
                updateLightAngle(mx - cx, my - cy, draggingLight);
                return true;
            }

            if (originalOnMouseDown) {
                return originalOnMouseDown(e, localPos);
            }
            return false;
        };

        const originalOnMouseMove = node.onMouseMove?.bind(node);
        node.onMouseMove = function (e, localPos) {
            if (draggingLight) {
                const width = node.size[0];
                const radarY = node.size[1] - RADAR_HEIGHT - 12;
                const cx = width / 2;
                const cy = radarY + RADAR_HEIGHT / 2;
                updateLightAngle(localPos[0] - cx, localPos[1] - cy, draggingLight);
                return true;
            }
            if (originalOnMouseMove) {
                return originalOnMouseMove(e, localPos);
            }
            return false;
        };

        const originalOnMouseUp = node.onMouseUp?.bind(node);
        node.onMouseUp = function (e, localPos) {
            if (draggingLight) {
                draggingLight = null;
                return true;
            }
            if (originalOnMouseUp) {
                return originalOnMouseUp(e, localPos);
            }
            return false;
        };

        function updateLightAngle(dx, dy, widgetName) {
            const w = getWidget(widgetName);
            if (!w) return;
            // Camera at bottom: dy > 0 is bottom (0°), dx > 0 is right (+90°), dy < 0 is top (+/-180°)
            let deg = Math.atan2(dx, dy) * (180 / Math.PI);
            deg = Math.round(deg);
            w.value = deg;
            if (w.inputEl) w.inputEl.value = deg;
            if (presetWidget && presetWidget.value !== "Custom") {
                presetWidget.value = "Custom";
                if (presetWidget.inputEl) presetWidget.inputEl.value = "Custom";
            }
            if (w.callback) {
                w.callback(w.value);
            }
            node.setDirtyCanvas(true, true);
            app.canvas?.setDirty(true, true);
        }

        // Draw interactive visual studio radar on node canvas
        const originalOnDrawForeground = node.onDrawForeground?.bind(node);
        node.onDrawForeground = function (ctx) {
            if (originalOnDrawForeground) {
                originalOnDrawForeground(ctx);
            }

            if (node.flags?.collapsed) return;

            const width = node.size[0];
            const radarY = node.size[1] - RADAR_HEIGHT - 12;
            const cx = width / 2;
            const cy = radarY + RADAR_HEIGHT / 2;
            const radius = Math.min(width, RADAR_HEIGHT) / 2 - 20;

            ctx.save();

            // Background pod container
            ctx.fillStyle = "#121215";
            ctx.strokeStyle = "#2c2c34";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.roundRect(14, radarY, width - 28, RADAR_HEIGHT, 8);
            ctx.fill();
            ctx.stroke();

            // Circular radar grid
            ctx.strokeStyle = "#222228";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(cx, cy, radius, 0, Math.PI * 2);
            ctx.stroke();

            ctx.beginPath();
            ctx.arc(cx, cy, radius * 0.5, 0, Math.PI * 2);
            ctx.stroke();

            // Crosshairs
            ctx.strokeStyle = "#1e1e24";
            ctx.beginPath();
            ctx.moveTo(cx - radius, cy);
            ctx.lineTo(cx + radius, cy);
            ctx.moveTo(cx, cy - radius);
            ctx.lineTo(cx, cy + radius);
            ctx.stroke();

            // Diagonal guidelines (+/- 45°, +/- 135°)
            ctx.strokeStyle = "#1a1a20";
            for (const angle of [45, 135, -45, -135]) {
                const rad = angle * (Math.PI / 180);
                ctx.beginPath();
                ctx.moveTo(cx, cy);
                ctx.lineTo(cx + radius * Math.sin(rad), cy + radius * Math.cos(rad));
                ctx.stroke();
            }

            // Elevation readout in top-left corner
            const kEl = getWidget("key_elevation")?.value ?? 35;
            const fEl = getWidget("fill_elevation")?.value ?? 15;
            const rEl = getWidget("rim_elevation")?.value ?? 40;
            ctx.fillStyle = "#6e6e7c";
            ctx.font = "8px monospace";
            ctx.textAlign = "left";
            ctx.fillText(`EL: K:${Math.round(kEl)}° F:${Math.round(fEl)}° R:${Math.round(rEl)}°`, 22, radarY + 14);

            // Camera indicator at bottom (0°) - User is behind the camera
            ctx.fillStyle = "#9a9aa8";
            ctx.font = "bold 8px monospace";
            ctx.textAlign = "center";
            ctx.fillText("▼ CAMERA (0°)", cx, cy + radius + 14);

            // Center subject dot
            ctx.fillStyle = "#ffffff";
            ctx.beginPath();
            ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
            ctx.fill();

            // Helper to render light ray & handle
            function drawLightHandle(azimuth, elevation, color, label, enabled) {
                if (!enabled) return;

                // 0° is bottom (camera), +45° is camera-right (screen right), +/-180° is top (backlight)
                const rad = azimuth * (Math.PI / 180);
                const lx = cx + radius * Math.sin(rad);
                const ly = cy + radius * Math.cos(rad);

                // Ray line
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(cx, cy);
                ctx.lineTo(lx, ly);
                ctx.stroke();

                // Glowing handle
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(lx, ly, 5.5, 0, Math.PI * 2);
                ctx.fill();

                ctx.strokeStyle = "#ffffff";
                ctx.lineWidth = 1.2;
                ctx.stroke();

                // Text badge
                ctx.fillStyle = color;
                ctx.font = "bold 9px sans-serif";
                const isRight = Math.sin(rad) >= 0;
                ctx.textAlign = isRight ? "left" : "right";
                ctx.fillText(`${label} ${Math.round(azimuth)}°`, lx + (isRight ? 8 : -8), ly + 3);
            }

            const kAz = getWidget("key_azimuth")?.value ?? 45;
            const kEn = getWidget("key_enabled")?.value ?? true;

            const fAz = getWidget("fill_azimuth")?.value ?? -30;
            const fEn = getWidget("fill_enabled")?.value ?? true;

            const rAz = getWidget("rim_azimuth")?.value ?? 140;
            const rEn = getWidget("rim_enabled")?.value ?? true;

            // Draw lights
            drawLightHandle(kAz, kEl, "#ffb92d", "KEY", kEn);
            drawLightHandle(fAz, fEl, "#41cdff", "FILL", fEn);
            drawLightHandle(rAz, rEl, "#f54baf", "RIM", rEn);

            ctx.restore();
        };
    }
});
