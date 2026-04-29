import React from "react";
import { createRoot } from "react-dom/client";

import { App } from "./app/App";
import { DebugConsoleApp } from "./app/DebugConsoleApp";

const params = new URLSearchParams(window.location.search);
const RootApp = params.get("debugConsole") === "1" ? DebugConsoleApp : App;

createRoot(document.getElementById("root")).render(<RootApp />);
