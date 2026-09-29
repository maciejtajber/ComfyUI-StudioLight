/**
 * ComfyUI-StudioLight — Studio3DStage Frontend Controller
 * Author: Maciej Tajber (https://github.com/maciejtajber)
 * License: MIT
 *
 * Interactive 3D Mannequin Viewport, Orbit Controls, Fashion Pose Presets,
 * and FK Joint Rotation Modal on the node face.
 */

import { app } from "../../scripts/app.js";
import * as THREE from "./vendor/three/three.module.min.js";
import { OrbitControls } from "./vendor/three/OrbitControls.js";
import { TransformControls } from "./vendor/three/TransformControls.js";
import { GLTFLoader } from "./vendor/three/GLTFLoader.js";

// Canonical fashion pose definitions (Euler angles in radians: [x, y, z])
const POSE_LIBRARY = {
    "Standing Fashion": {
        "mixamorig:Spine": [-0.05, 0.05, -0.04],
        "mixamorig:Spine1": [0.03, 0.04, -0.02],
        "mixamorig:Neck": [0.04, -0.12, 0.03],
        "mixamorig:Head": [0.05, -0.15, 0.02],
        "mixamorig:LeftArm": [0.25, 0.1, -0.3],
        "mixamorig:RightArm": [0.15, -0.05, 0.25],
        "mixamorig:LeftForeArm": [0.4, 0.0, 0.1],
        "mixamorig:RightForeArm": [0.3, 0.0, -0.1],
        "mixamorig:RightUpLeg": [0.05, 0.05, 0.08],
        "mixamorig:LeftUpLeg": [-0.05, -0.05, -0.05],
    },
    "American Plan (Hand on Hip)": {
        "mixamorig:Spine": [-0.06, 0.1, -0.06],
        "mixamorig:Spine1": [0.05, 0.05, -0.03],
        "mixamorig:Neck": [0.05, -0.18, 0.04],
        "mixamorig:Head": [0.08, -0.22, 0.02],
        "mixamorig:RightArm": [0.35, 0.2, 0.75],
        "mixamorig:RightForeArm": [1.45, 0.35, -0.2],
        "mixamorig:LeftArm": [0.2, 0.05, -0.25],
        "mixamorig:LeftForeArm": [0.35, 0.0, 0.05],
        "mixamorig:RightUpLeg": [0.1, 0.08, 0.12],
    },
    "Sitting Editorial": {
        "mixamorig:Hips": [0.0, 0.0, 0.0],
        "mixamorig:Spine": [0.15, -0.05, 0.0],
        "mixamorig:LeftUpLeg": [-1.45, 0.2, -0.2],
        "mixamorig:RightUpLeg": [-1.35, -0.15, 0.25],
        "mixamorig:LeftLeg": [1.45, 0.0, 0.0],
        "mixamorig:RightLeg": [1.35, 0.0, 0.0],
        "mixamorig:LeftArm": [0.4, 0.2, -0.3],
        "mixamorig:RightArm": [0.5, -0.2, 0.35],
    },
    "3/4 Head Turn": {
        "mixamorig:Spine": [-0.04, 0.12, -0.02],
        "mixamorig:Neck": [0.06, -0.45, 0.05],
        "mixamorig:Head": [0.12, -0.55, 0.02],
        "mixamorig:LeftArm": [0.2, 0.05, -0.2],
        "mixamorig:RightArm": [0.2, -0.05, 0.2],
    },
    "Athletic Stance": {
        "mixamorig:Spine": [0.08, 0.0, 0.0],
        "mixamorig:LeftUpLeg": [-0.15, 0.1, -0.3],
        "mixamorig:RightUpLeg": [0.15, -0.1, 0.3],
        "mixamorig:LeftArm": [0.4, 0.2, -0.45],
        "mixamorig:RightArm": [0.4, -0.2, 0.45],
        "mixamorig:LeftForeArm": [0.6, 0.0, 0.0],
        "mixamorig:RightForeArm": [0.6, 0.0, 0.0],
    },
    "Crouched Fashion": {
        "mixamorig:Hips": [0.0, 0.0, 0.0],
        "mixamorig:Spine": [0.35, 0.0, 0.0],
        "mixamorig:LeftUpLeg": [-1.8, 0.3, -0.2],
        "mixamorig:RightUpLeg": [-1.9, -0.2, 0.3],
        "mixamorig:LeftLeg": [2.3, 0.0, 0.0],
        "mixamorig:RightLeg": [2.4, 0.0, 0.0],
        "mixamorig:LeftArm": [0.7, 0.1, -0.2],
        "mixamorig:RightArm": [0.7, -0.1, 0.2],
    }
};

class Stage3DViewport {
    constructor(node) {
        this.node = node;
        this.width = 340;
        this.height = 340;
        this.locked = false;
        this.activeModel = "Female Mannequin";
        this.activePose = "Standing Fashion";

        // 1. Offscreen WebGL Canvas & Renderer
        this.canvas = document.createElement("canvas");
        this.canvas.width = this.width;
        this.canvas.height = this.height;

        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: true,
            preserveDrawingBuffer: true,
        });
        this.renderer.setSize(this.width, this.height, false);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

        // 2. Three.js Scene & Camera
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x18181b);

        this.camera = new THREE.PerspectiveCamera(40, 1.0, 0.1, 50);
        this.camera.position.set(0, 1.25, 2.3);
        this.target = new THREE.Vector3(0, 1.05, 0);
        this.camera.lookAt(this.target);

        // 3. Stage Studio Lighting
        const keyLight = new THREE.DirectionalLight(0xfff5e6, 1.8);
        keyLight.position.set(2.0, 2.5, 2.0);
        this.scene.add(keyLight);

        const fillLight = new THREE.DirectionalLight(0xd0e8ff, 0.65);
        fillLight.position.set(-2.0, 1.2, 1.5);
        this.scene.add(fillLight);

        const rimLight = new THREE.DirectionalLight(0xffffff, 0.9);
        rimLight.position.set(0.0, 2.5, -2.0);
        this.scene.add(rimLight);

        const ambLight = new THREE.AmbientLight(0x222226, 1.0);
        this.scene.add(ambLight);

        // Studio Floor Grid
        const grid = new THREE.GridHelper(4, 16, 0x3f3f46, 0x27272a);
        grid.position.y = 0;
        this.scene.add(grid);

        // 4. Mannequin Material (Neutral White Plaster Clay)
        this.studioMaterial = new THREE.MeshStandardMaterial({
            color: 0xe4e4e7,
            roughness: 0.5,
            metalness: 0.05,
        });

        this.bones = {};
        this.currentMeshGroup = null;
        this.loader = new GLTFLoader();

        // Mouse drag tracking for orbit
        this.isDragging = false;
        this.dragButton = 0;
        this.lastMouseX = 0;
        this.lastMouseY = 0;

        // Load default model
        this.loadModel(this.activeModel);
    }

    loadModel(modelName) {
        const file = modelName.includes("Male") ? "mannequin_male.glb" : "mannequin_female.glb";
        const url = `/studiolight/assets/${file}`;

        this.loader.load(
            url,
            (gltf) => {
                if (this.currentMeshGroup) {
                    this.scene.remove(this.currentMeshGroup);
                }

                this.currentMeshGroup = gltf.scene;
                this.bones = {};

                // Normalize scale and position
                const box = new THREE.Box3().setFromObject(gltf.scene);
                const size = box.getSize(new THREE.Vector3());
                const maxDim = Math.max(size.x, size.y, size.z);
                const scale = 1.8 / (maxDim || 1.8);
                gltf.scene.scale.set(scale, scale, scale);

                box.setFromObject(gltf.scene);
                const center = box.getCenter(new THREE.Vector3());
                gltf.scene.position.x = -center.x;
                gltf.scene.position.y = -box.min.y;
                gltf.scene.position.z = -center.z;

                // Index bones and apply neutral plaster material
                gltf.scene.traverse((child) => {
                    if (child.isBone) {
                        this.bones[child.name] = child;
                    }
                    if (child.isMesh) {
                        child.material = this.studioMaterial;
                        child.castShadow = true;
                        child.receiveShadow = true;
                    }
                });

                this.scene.add(gltf.scene);
                this.applyPose(this.activePose);
                this.render();
                if (this.node) this.node.setDirtyCanvas(true, true);
            },
            undefined,
            (err) => console.error("[Studio3DStage] Error loading model:", err)
        );
    }

    applyPose(poseName) {
        this.activePose = poseName;
        const pose = POSE_LIBRARY[poseName] || POSE_LIBRARY["Standing Fashion"];
        if (!pose || !this.bones) return;

        // Reset all bones to identity rotation first
        for (const bone of Object.values(this.bones)) {
            bone.rotation.set(0, 0, 0);
        }

        // Apply target joint Euler angles
        for (const [boneName, rot] of Object.entries(pose)) {
            const bone = this.bones[boneName];
            if (bone) {
                bone.rotation.set(rot[0], rot[1], rot[2]);
            }
        }

        if (this.currentMeshGroup) {
            this.currentMeshGroup.updateMatrixWorld(true);
        }
        this.render();
    }

    render() {
        this.renderer.render(this.scene, this.camera);
    }

    handleMouseDown(event, localX, localY) {
        if (this.locked) return false;
        this.isDragging = true;
        this.dragButton = event.button || 0;
        this.lastMouseX = localX;
        this.lastMouseY = localY;
        return true;
    }

    handleMouseMove(event, localX, localY) {
        if (!this.isDragging || this.locked) return false;

        const dx = localX - this.lastMouseX;
        const dy = localY - this.lastMouseY;
        this.lastMouseX = localX;
        this.lastMouseY = localY;

        if (this.dragButton === 0) {
            // Left click: Orbit around target
            const offset = this.camera.position.clone().sub(this.target);
            let radius = offset.length();
            let theta = Math.atan2(offset.x, offset.z);
            let phi = Math.acos(Math.min(Math.max(offset.y / radius, -1), 1));

            theta -= dx * 0.015;
            phi = Math.max(0.1, Math.min(Math.PI - 0.1, phi - dy * 0.015));

            offset.x = radius * Math.sin(phi) * Math.sin(theta);
            offset.y = radius * Math.cos(phi);
            offset.z = radius * Math.sin(phi) * Math.cos(theta);

            this.camera.position.copy(this.target).add(offset);
            this.camera.lookAt(this.target);
        } else if (this.dragButton === 2) {
            // Right click: Pan camera vertically and horizontally
            const panSpeed = 0.003;
            const right = new THREE.Vector3();
            const up = new THREE.Vector3(0, 1, 0);
            this.camera.getWorldDirection(right);
            right.cross(up).normalize();

            this.camera.position.addScaledVector(right, -dx * panSpeed);
            this.target.addScaledVector(right, -dx * panSpeed);

            this.camera.position.y += dy * panSpeed;
            this.target.y += dy * panSpeed;
        }

        this.render();
        return true;
    }

    handleMouseUp() {
        this.isDragging = false;
    }

    handleWheel(event) {
        if (this.locked) return false;
        const zoomDelta = event.deltaY * 0.0025;
        const offset = this.camera.position.clone().sub(this.target);
        const dist = offset.length();
        const newDist = Math.max(0.6, Math.min(6.0, dist * (1 + zoomDelta)));
        offset.setLength(newDist);
        this.camera.position.copy(this.target).add(offset);
        this.render();
        return true;
    }
}

function hookStageNode(node) {
    if (!node || node._sl_stage_hooked) return;
    node._sl_stage_hooked = true;

    const viewport = new Stage3DViewport(node);
    node._stageViewport = viewport;

    // 1. Add Custom Interactive 3D Viewport Widget
    const VIEWPORT_SIZE = 340;
    const viewportWidget = {
        type: "custom",
        name: "studio_3d_viewport",
        value: "",
        options: { serialize: false },
        computeSize: (width) => [width, VIEWPORT_SIZE + 36],
        draw: function (ctx, n, width, y, height) {
            ctx.save();
            const margin = 10;
            const drawW = width - margin * 2;
            const drawH = VIEWPORT_SIZE;
            const drawX = margin;
            const drawY = y + 4;

            // Frame background
            ctx.fillStyle = "#121214";
            ctx.strokeStyle = "#3f3f46";
            ctx.lineWidth = 1.0;
            ctx.beginPath();
            if (ctx.roundRect) ctx.roundRect(drawX, drawY, drawW, drawH, 6);
            else ctx.rect(drawX, drawY, drawW, drawH);
            ctx.fill();
            ctx.stroke();

            // Draw WebGL frame
            if (viewport.canvas) {
                ctx.drawImage(viewport.canvas, drawX + 2, drawY + 2, drawW - 4, drawH - 4);
            }

            // Lock Indicator Badge & Controls
            const lockY = drawY + drawH + 6;
            const lockH = 22;
            ctx.fillStyle = viewport.locked ? "rgba(239, 68, 68, 0.15)" : "rgba(34, 197, 94, 0.15)";
            ctx.strokeStyle = viewport.locked ? "#ef4444" : "#22c55e";
            ctx.beginPath();
            if (ctx.roundRect) ctx.roundRect(drawX, lockY, 110, lockH, 4);
            else ctx.rect(drawX, lockY, 110, lockH);
            ctx.fill();
            ctx.stroke();

            ctx.font = "bold 9px sans-serif";
            ctx.fillStyle = viewport.locked ? "#f87171" : "#4ade80";
            ctx.textAlign = "center";
            ctx.fillText(viewport.locked ? "🔒 CAMERA LOCKED" : "🔓 CAMERA FREE", drawX + 55, lockY + lockH / 2 + 1);

            // Orbit Hint
            ctx.fillStyle = "#71717a";
            ctx.font = "9px sans-serif";
            ctx.textAlign = "right";
            ctx.fillText("Left: Orbit | Right: Pan | Scroll: Zoom", drawX + drawW, lockY + lockH / 2 + 1);

            ctx.restore();
        },
        mouse: function (event, pos, n) {
            const localX = pos[0] - 10;
            const localY = pos[1] - (this.last_y || 0);

            // Clicked Lock Button?
            if (localY >= VIEWPORT_SIZE + 6 && localY <= VIEWPORT_SIZE + 32 && localX <= 110) {
                if (event.type === "pointerdown" || event.type === "mousedown") {
                    viewport.locked = !viewport.locked;
                    n.setDirtyCanvas(true, true);
                    return true;
                }
            }

            // In 3D Canvas Area
            if (localY >= 4 && localY <= VIEWPORT_SIZE + 4) {
                if (event.type === "pointerdown" || event.type === "mousedown") {
                    return viewport.handleMouseDown(event, localX, localY);
                } else if (event.type === "pointermove" || event.type === "mousemove") {
                    return viewport.handleMouseMove(event, localX, localY);
                } else if (event.type === "pointerup" || event.type === "mouseup") {
                    viewport.handleMouseUp();
                    return true;
                } else if (event.type === "wheel") {
                    return viewport.handleWheel(event);
                }
            }
            return false;
        }
    };

    // Insert Viewport at top of widgets
    node.widgets.unshift(viewportWidget);

    // 2. Hook Model & Pose Preset Dropdown Changes
    const modelWidget = node.widgets.find((w) => w.name === "model_preset");
    if (modelWidget) {
        const origModelCb = modelWidget.callback;
        modelWidget.callback = function (val) {
            if (origModelCb) origModelCb.apply(this, arguments);
            const m = typeof val === "string" ? val : this.value;
            viewport.loadModel(m);
        };
    }

    const poseWidget = node.widgets.find((w) => w.name === "pose_preset");
    if (poseWidget) {
        const origPoseCb = poseWidget.callback;
        poseWidget.callback = function (val) {
            if (origPoseCb) origPoseCb.apply(this, arguments);
            const p = typeof val === "string" ? val : this.value;
            viewport.applyPose(p);
        };
    }

    // Auto-adjust node size
    const sz = node.computeSize();
    node.setSize([Math.max(node.size[0], 360), sz[1]]);
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "ComfyUI.StudioLight.3DStage",

    nodeCreated(node) {
        if (node.comfyClass === "Studio3DStage" || node.type === "Studio3DStage") {
            hookStageNode(node);
        }
    },

    loadedGraphNode(node) {
        if (node.comfyClass === "Studio3DStage" || node.type === "Studio3DStage") {
            hookStageNode(node);
        }
    }
});
