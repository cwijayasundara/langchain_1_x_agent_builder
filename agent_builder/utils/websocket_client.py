"""
WebSocket Client for streaming agent responses in real-time.
Handles async WebSocket connections with threading support for Streamlit.
"""

import asyncio
import websockets
import json
from typing import Callable, Optional, Dict, Any, List
from queue import Queue
import threading


class WebSocketClient:
    """WebSocket client for streaming agent responses."""

    def __init__(self, base_url: str):
        """
        Initialize WebSocket client.

        Args:
            base_url: Base URL of the API (http/https will be converted to ws/wss)
        """
        # Convert HTTP URL to WebSocket URL
        self.base_url = base_url.replace('http://', 'ws://').replace('https://', 'wss://').rstrip('/')
        self.ws = None
        self.connected = False

    async def connect(self, agent_id: str):
        """
        Establish WebSocket connection to agent stream endpoint.

        Args:
            agent_id: Agent identifier

        Raises:
            Exception: If connection fails
        """
        uri = f"{self.base_url}/execution/{agent_id}/stream"
        try:
            self.ws = await websockets.connect(uri)
            self.connected = True
        except Exception as e:
            self.connected = False
            raise Exception(f"Failed to connect to WebSocket: {str(e)}")

    async def send_message(
        self,
        messages: List[Dict[str, str]],
        thread_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Send message to agent via WebSocket.

        Args:
            messages: List of message dicts with role and content
            thread_id: Optional thread ID for conversation continuity
            context: Optional runtime context values

        Raises:
            Exception: If not connected or send fails
        """
        if not self.connected or not self.ws:
            raise Exception("WebSocket not connected")

        data = {
            "messages": messages
        }

        if thread_id:
            data["thread_id"] = thread_id

        if context:
            data["context"] = context

        await self.ws.send(json.dumps(data))

    async def receive_stream(
        self,
        on_chunk: Callable[[str], None],
        on_complete: Callable[[str], None],
        on_error: Callable[[Dict], None]
    ):
        """
        Receive streaming response from agent.

        Args:
            on_chunk: Callback for each chunk received (chunk_data)
            on_complete: Callback when stream completes (thread_id)
            on_error: Callback for errors (error_dict)

        Raises:
            Exception: If not connected or receive fails
        """
        if not self.connected or not self.ws:
            raise Exception("WebSocket not connected")

        try:
            async for message in self.ws:
                data = json.loads(message)

                msg_type = data.get("type")

                if msg_type == "chunk":
                    on_chunk(data.get("data", ""))

                elif msg_type == "complete":
                    on_complete(data.get("thread_id", ""))
                    break

                elif msg_type == "error":
                    on_error(data.get("error", {}))
                    break

        except Exception as e:
            on_error({"code": "STREAM_ERROR", "message": str(e)})

    async def close(self):
        """Close WebSocket connection."""
        if self.ws:
            await self.ws.close()
            self.connected = False


class StreamingHandler:
    """
    Thread-safe streaming handler for use with Streamlit.
    Runs WebSocket client in background thread and communicates via Queue.
    """

    def __init__(self, base_url: str):
        """
        Initialize streaming handler.

        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url
        self.chunk_queue = Queue()
        self.error = None
        self.thread_id = None
        self.completed = False

    def stream_message(
        self,
        agent_id: str,
        messages: List[Dict[str, str]],
        thread_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Stream a message to the agent in a background thread.

        Args:
            agent_id: Agent identifier
            messages: List of message dicts
            thread_id: Optional thread ID
            context: Optional context values
        """
        def websocket_task():
            """Task to run WebSocket client asynchronously."""
            asyncio.run(self._stream_async(agent_id, messages, thread_id, context))

        thread = threading.Thread(target=websocket_task, daemon=True)
        thread.start()

    async def _stream_async(
        self,
        agent_id: str,
        messages: List[Dict],
        thread_id: Optional[str],
        context: Optional[Dict]
    ):
        """
        Async method to handle WebSocket streaming.

        Args:
            agent_id: Agent identifier
            messages: Message list
            thread_id: Optional thread ID
            context: Optional context
        """
        client = WebSocketClient(self.base_url)

        try:
            # Connect
            await client.connect(agent_id)

            # Send message
            await client.send_message(messages, thread_id, context)

            # Receive stream
            await client.receive_stream(
                on_chunk=self._on_chunk,
                on_complete=self._on_complete,
                on_error=self._on_error
            )

        except Exception as e:
            self._on_error({"code": "CONNECTION_ERROR", "message": str(e)})

        finally:
            await client.close()

    def _on_chunk(self, chunk: str):
        """Handle incoming chunk."""
        self.chunk_queue.put(("chunk", chunk))

    def _on_complete(self, thread_id: str):
        """Handle stream completion."""
        self.thread_id = thread_id
        self.completed = True
        self.chunk_queue.put(("complete", thread_id))

    def _on_error(self, error: Dict):
        """Handle error."""
        self.error = error
        self.chunk_queue.put(("error", error))

    def get_chunk(self, timeout: float = 0.1) -> Optional[tuple]:
        """
        Get next chunk from queue (non-blocking).

        Args:
            timeout: Timeout in seconds

        Returns:
            Tuple of (type, data) or None if queue empty
        """
        try:
            return self.chunk_queue.get(timeout=timeout)
        except:
            return None

    def has_chunks(self) -> bool:
        """Check if there are chunks available."""
        return not self.chunk_queue.empty()

    def is_complete(self) -> bool:
        """Check if streaming is complete."""
        return self.completed

    def get_error(self) -> Optional[Dict]:
        """Get error if one occurred."""
        return self.error
