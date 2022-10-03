import { Server } from "http";
import WebSocket from "ws";
import { verifyConnection } from "./auth";

export default (expressServer: Server) => {
  const websocketServer = new WebSocket.Server({
    noServer: true,
    path: "/websockets",
  });

  expressServer.on("upgrade", async (request, socket, head) => {
    const userInfo = await verifyConnection(request);
    if (userInfo) {
      websocketServer.handleUpgrade(request, socket, head, (websocket) => {
        websocketServer.emit("connection", websocket, request);
      });
    } else {
      socket.destroy();
    }
  });
};
