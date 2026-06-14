import { onUnmounted } from "vue";
import { useSituationStore } from "../stores/situation";
import { useSimulation } from "./useSimulation";

const DEMO_TIMEOUT_MS = 2500;

function buildWebSocketUrl() {
  const explicitUrl = import.meta.env.VITE_WS_URL;
  if (explicitUrl) {
    return explicitUrl;
  }

  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const { hostname, port } = window.location;
  const host = port ? `${hostname}:${port}` : hostname;
  return `${protocol}//${host}/ws/traffic`;
}

export function useWebSocket() {
  const store = useSituationStore();
  const { startSimulation, stopSimulation } = useSimulation();
  let ws = null;
  let reconnectTimer = null;
  let connectTimeout = null;
  let manualClose = false;
  let demoFallback = false;

  const clearTimers = () => {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    if (connectTimeout) {
      clearTimeout(connectTimeout);
      connectTimeout = null;
    }
  };

  const enableDemoMode = () => {
    if (demoFallback) {
      return;
    }

    demoFallback = true;
    manualClose = true;
    clearTimers();

    if (ws) {
      ws.close();
      ws = null;
    }

    startSimulation();
  };

  const connect = () => {
    const forceDemo =
      new URLSearchParams(window.location.search).get("demo") === "1";
    if (forceDemo) {
      enableDemoMode();
      return;
    }

    manualClose = false;
    demoFallback = false;
    clearTimers();

    const wsUrl = buildWebSocketUrl();

    try {
      ws = new WebSocket(wsUrl);
      connectTimeout = setTimeout(() => {
        store.setConnectionStatus("demo");
        enableDemoMode();
      }, DEMO_TIMEOUT_MS);

      ws.onopen = () => {
        clearTimers();
        if (store.simulationActive) {
          stopSimulation();
        }
        store.setConnectionStatus("connected");
        store.startTime = Date.now();
        console.log("WebSocket connected");
      };

      ws.onclose = () => {
        clearTimers();
        if (manualClose || demoFallback) {
          return;
        }
        store.setConnectionStatus("disconnected");
        console.log("WebSocket disconnected, reconnecting...");
        reconnectTimer = setTimeout(connect, 3000);
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        if (!demoFallback) {
          store.setConnectionStatus("error");
        }
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          store.processRealtimeData(data);
        } catch (e) {
          console.error("Data parse error:", e);
        }
      };
    } catch (e) {
      console.error("WebSocket connection failed:", e);
      store.setConnectionStatus("demo");
      enableDemoMode();
    }
  };

  const disconnect = () => {
    manualClose = true;
    clearTimers();
    if (ws) {
      ws.close();
      ws = null;
    }
  };

  onUnmounted(() => {
    disconnect();
  });

  return {
    connect,
    disconnect,
  };
}
