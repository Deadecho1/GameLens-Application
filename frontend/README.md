
# 🎮 GameLens | Advanced Gaming Analytics Dashboard

**GameLens** is a cutting-edge analytics platform designed for game developers to derive deep insights from gameplay footage. Using Computer Vision (YOLOv12, SAM2, and OCR), it transforms raw video into actionable data, presented through a high-fidelity "Gamer-HUD" interface.

---

## 💎 The Central Intelligence Hub (`dataStore.js`)

The entire application operates on a **Single Source of Truth** architecture. All dynamic data flows through a central bridge:

> ### 📂 `src/dataStore.js`
> **This is the most critical file for Backend Integration.**
>
> - **Input (Backend to UI):** Any data processed by the CV models (Boss names, Item detection, Run durations, Terminal logs) must be injected into the `initialData` object in this file.
> - **Output (UI to Backend):** User selections (Game versions, chosen clips) are tracked here to be sent back for processing.
> - **Sync:** The UI components are "reactive"—as soon as you update this JSON structure, all charts, tables, and status indicators will update in real-time.

---

## 🚀 Mission Briefing (Quick Start)

Follow these steps to deploy the dashboard locally:

### 1. Environmental Sync
Ensure you have **Node.js** (v20.14.0 or higher) installed on your system.

### 2. Tactical Installation
Navigate to the frontend directory and install the complete Tech Stack (React, Tailwind v4, Charts, and Animations) with a single command:
```bash
cd frontend
npm install
```

### 3. System Launch
Initialize the Vite development server:
```bash
npm run dev
```
**Access Point:** `http://localhost:5173`

---

## 🛠 Tech Stack (The Arsenal)

The dashboard is built using state-of-the-art web technologies:
* **Core:** [React](https://reactjs.org/) (Vite)
* **Styling:** [Tailwind CSS v4](https://tailwindcss.com/) (Custom Glassmorphism & Neon UI)
* **Data Viz:** [Recharts](https://recharts.org/) (Dynamic Boss & Item Analytics)
* **Motion:** [Framer Motion](https://www.framer.com/motion/) (Fluid "Mission-Style" transitions)
* **Icons:** [Lucide-React](https://lucide.dev/) (Tactical iconography)

---

## 📂 System Architecture

* **`src/dataStore.js`**: The Brain – Manages all global states and mock/real data.
* **`src/App.jsx`**: Mission Control – Handles the 3-step workflow logic.
* **`src/components/`**: Modular HUD components (Synergy Lab, Survival Charts, Sidebar).
* **`src/index.css`**: Global Styles – Contains the "Scanline" and "Neon" CSS effects.

---

## 🎯 Key Modules

### 🛠 Mission Setup & Process
A unified 3-step workflow (`Configure -> Initialize -> Execute`) that guides the user from selecting a game version to running the processing pipeline with real-time log monitoring.

### 💀 Boss Intelligence
A master-detail interface providing global survival statistics for each boss. Includes the **Item Synergy Analyzer** to identify which equipment combinations are most lethal.

### 🧪 Item Power Lab
An interactive survival simulator. Build a "Loadout" of up to 5 items to calculate their impact on average run duration, helping developers balance the game's "Meta."

### 📈 Global Command Center
High-level overview of system performance, including run duration trends, efficiency scores, and game-wide statistics.


# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react/README.md) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh
