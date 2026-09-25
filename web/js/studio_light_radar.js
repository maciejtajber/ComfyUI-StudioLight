/**
 * ComfyUI-StudioLight — Interactive 2D Studio Radar & Angle Visualizer
 * Author: Maciej Tajber (https://github.com/maciejtajber)
 * License: MIT
 *
 * Renders a tactile, real-time photographic studio compass directly on the node canvas face.
 * Visualizes Key, Fill, and Rim azimuth and elevation vectors and supports click-and-drag aiming.
 */

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI.StudioLight.Radar",

    async nodeCreated(node) {
        if (node.comfyClass !== "StudioLightRig") {
            return;
        }

        // Dedicated height allocated for the radar visualization
        const RADAR_HEIGHT = 180;
        let draggingLight = null;

        // Ensure node has enough room
        const originalComputeSize = node.computeSize.bind(node);
        node.computeSize = function () {
            const size = originalComputeSize();
            size[1] = Math.max(size[1], 460 + RADAR_HEIGHT);
            size[0] = Math.max(size[0], 360);
            return size;
        };

        function getWidget(name) {
            return node.widgets?.find((w) => w.name === name);
        }

        // Intercept mouse clicks and drags on the radar area
        const originalOnMouseDown = node.onMouseDown?.bind(node);
        node.onMouseDown = function (e, localPos) {
            const width = node.size[0];
            const radarY = 50; // Placed right below node title/preset
            const cx = width / 2;
            const cy = radarY + RADAR_HEIGHT / 2;
            const radius = Math.min(width, RADAR_HEIGHT) / 2 - 20;

            const mx = localPos[0];
            const my = localPos[1];

            // Check if click is inside radar circle
            const dist = Math.hypot(mx - cx, my - cy);
            if (dist <= radius + 15 && my >= radarY && my <= radarY + RADAR_HEIGHT) {
                // Determine which light head was clicked
                const lights = [
                    { id: "key", widget: "key_azimuth", col: [255, 185, 45] },
                    { id: "fill", widget: "fill_azimuth", col: [65, 205, 255] },
                    { id: "rim", widget: "rim_azimuth", col: [245, 75, 175] },
                ];

                for (const l of lights) {
                    const w = getWidget(l.widget);
                    if (!w) continue;
                    const az = (w.value - 90) * (Math.PI / 180);
                    const lx = cx + radius * Math.cos(az);
                    const ly = cy + radius * Math.sin(az);
                    if (Math.hypot(mx - lx, my - ly) <= 18) {
                        draggingLight = l.widget;
                        return true;
                    }
                }

                // If clicked anywhere in radar, target Key light by default
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
                const radarY = 50;
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
            // Angle where 0° is Top (-Y), +90° is Right (+X), -90° is Left (-X)
            let deg = Math.atan2(dx, -dy) * (180 / Math.PI);
            deg = Math.round(deg);
            w.value = deg;
            if (w.callback) {
                w.callback(w.value);
            }
            node.setDirtyCanvas(true, true);
        }

        // Draw interactive visual studio radar on node canvas
        const originalOnDrawForeground = node.onDrawForeground?.bind(node);
        node.onDrawForeground = function (ctx) {
            if (originalOnDrawForeground) {
                originalOnDrawForeground(ctx);
            }

            if (node.flags.collapsed) return;

            const width = node.size[0];
            const radarY = 50;
            const cx = width / 2;
            const cy = radarY + RADAR_HEIGHT / 2;
            const radius = Math.min(width, RADAR_HEIGHT) / 2 - 22;

            ctx.save();

            // Background pod container
            ctx.fillStyle = "#121214";
            ctx.strokeStyle = "#2c2c34";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.roundRect(14, radarY, width - 28, RADAR_HEIGHT - 6, 8);
            ctx.fill();
            ctx.stroke();

            // Circular radar grid
            ctx.strokeStyle = "#26262e";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(cx, cy, radius, 0, Math.PI * 2);
            ctx.stroke();

            ctx.beginPath();
            ctx.arc(cx, cy, radius * 0.5, 0, Math.PI * 2);
            ctx.stroke();

            // Crosshairs
            ctx.strokeStyle = "#222228";
            ctx.beginPath();
            ctx.moveTo(cx - radius, cy);
            ctx.lineTo(cx + radius, cy);
            ctx.moveTo(cx, cy - radius);
            ctx.lineTo(cx, cy + radius);
            ctx.stroke();

            // Camera indicator at top (0°)
            ctx.fillStyle = "#8e8e9a";
            ctx.font = "9px monospace";
            ctx.textAlign = "center";
            ctx.fillText("▲ CAMERA (0°)", cx, cy - radius - 6);

            // Center subject dot
            ctx.fillStyle = "#e0e0ea";
            ctx.beginPath();
            ctx.arc(cx, cy, 4, 0, Math.PI * 2);
            ctx.fill();

            // Helper to render light ray & handle
            function drawLightHandle(azimuth, elevation, color, label, enabled) {
                if (!enabled) return;

                const rad = (azimuth - 90) * (Math.PI / 180);
                const lx = cx + radius * Math.cos(rad);
                const ly = cy + radius * Math.sin(rad);

                // Ray line
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(cx, cy);
                ctx.lineTo(lx, ly);
                ctx.stroke();

                // Glow ring
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(lx, ly, 6, 0, Math.PI * 2);
                ctx.fill();

                ctx.strokeStyle = "#ffffff";
                ctx.lineWidth = 1.5;
                ctx.stroke();

                // Label badge
                ctx.fillStyle = color;
                ctx.font = "bold 9px sans-serif";
                const offset = Math.cos(rad) >= 0 ? 10 : -10;
                ctx.textAlign = Math.cos(rad) >= 0 ? "left" : "right";
                ctx.fillText(`${label} ${Math.round(azimuth)}°`, lx + offset, ly + 3);
            }

            const kAz = getWidget("key_azimuth")?.value ?? 45;
            const kEl = getWidget("key_elevation")?.value ?? 35;
            const kEn = getWidget("key_enabled")?.value ?? true;

            const fAz = getWidget("fill_azimuth")?.value ?? -30;
            const fEl = getWidget("fill_elevation")?.value ?? 15;
            const fEn = getWidget("fill_enabled")?.value ?? true;

            const rAz = getWidget("rim_azimuth")?.value ?? 160;
            const rEl = getWidget("rim_elevation")?.value ?? 45;
            const rEn = getWidget("rim_enabled")?.value ?? true;

            // Draw lights
            drawLightHandle(kAz, kEl, "#ffb92d", "KEY", kEn);
            drawLightHandle(fAz, fEl, "#41cdff", "FILL", fEn);
            drawLightHandle(rAz, rEl, "#f54baf", "RIM", rEn);

            // Elevation readout in bottom corner
            ctx.fillStyle = "#6e6e7c";
            ctx.font = "9px monospace";
            ctx.textAlign = "left";
            ctx.fillText(`EL: K:${kEl}° F:${fEl}° R:${rEl}°`, 22, radarY + RADAR_HEIGHT - 16);

            ctx.restore();
        };

        // Repaint when widgets change
        for (const w of node.widgets || []) {
            const originalCallback = w.callback;
            w.callback = function (val) {
                if (originalCallback) originalCallback.call(w, val);
                node.setDirtyCanvas(true, true);
            };
        }
    },
});
