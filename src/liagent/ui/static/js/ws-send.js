/**
 * WebSocket transport — zero-import module to break circular dependencies.
 * All modules that need to send WS messages or access the ws instance
 * import from here instead of ws-client.js.
 */

var ws = null;

export function getWs() { return ws; }
export function setWs(v) { ws = v; }

export function wsSend(payload) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(typeof payload === 'string' ? payload : JSON.stringify(payload));
  }
}
