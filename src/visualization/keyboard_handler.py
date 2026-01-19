"""WebSocket-based keyboard input handler for Viser.

Viser doesn't natively support keyboard input, so we run a separate WebSocket
server and inject JavaScript into the browser to capture key events.
"""
import asyncio
import json
import threading
import websockets
from typing import Optional, Callable
import logging

from ..core.constants import KEYBOARD_WS_PORT
from ..core.game_state import GameState

logger = logging.getLogger(__name__)


# JavaScript to inject into the browser for keyboard capture
KEYBOARD_JS = """
<script>
(function() {
    const ws = new WebSocket('ws://' + window.location.hostname + ':""" + str(KEYBOARD_WS_PORT) + """');

    ws.onopen = function() {
        console.log('Keyboard WebSocket connected');
    };

    ws.onerror = function(err) {
        console.error('Keyboard WebSocket error:', err);
    };

    document.addEventListener('keydown', function(e) {
        if (['w', 's', 'W', 'S', 'ArrowUp', 'ArrowDown'].includes(e.key)) {
            e.preventDefault();
            ws.send(JSON.stringify({type: 'keydown', key: e.key}));
        }
    });

    document.addEventListener('keyup', function(e) {
        if (['w', 's', 'W', 'S', 'ArrowUp', 'ArrowDown'].includes(e.key)) {
            e.preventDefault();
            ws.send(JSON.stringify({type: 'keyup', key: e.key}));
        }
    });

    // Keep connection alive
    setInterval(function() {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({type: 'ping'}));
        }
    }, 30000);
})();
</script>
"""


class KeyboardHandler:
    """Handles keyboard input via WebSocket."""

    def __init__(self, game_state: GameState, port: int = KEYBOARD_WS_PORT):
        """Initialize the keyboard handler.

        Args:
            game_state: The shared game state object.
            port: Port for the WebSocket server.
        """
        self.game_state = game_state
        self.port = port
        self._server: Optional[websockets.WebSocketServer] = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False

    async def _handle_client(self, websocket: websockets.WebSocketServerProtocol):
        """Handle messages from a connected client.

        Args:
            websocket: The WebSocket connection.
        """
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get('type')
                    key = data.get('key', '').lower()

                    # Normalize arrow keys
                    if key == 'arrowup':
                        key = 'w'
                    elif key == 'arrowdown':
                        key = 's'

                    if msg_type == 'keydown':
                        self.game_state.key_pressed(key)
                    elif msg_type == 'keyup':
                        self.game_state.key_released(key)
                    elif msg_type == 'ping':
                        pass  # Keep-alive, no action needed

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {message}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"WebSocket error: {e}")

    async def _run_server(self):
        """Run the WebSocket server."""
        self._server = await websockets.serve(
            self._handle_client,
            "0.0.0.0",
            self.port,
        )
        logger.info(f"Keyboard WebSocket server started on port {self.port}")
        await self._server.wait_closed()

    def _thread_main(self):
        """Main function for the keyboard handler thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(self._run_server())
        except Exception as e:
            logger.error(f"Keyboard server error: {e}")
        finally:
            self._loop.close()

    def start(self):
        """Start the keyboard handler in a background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        logger.info("Keyboard handler started")

    def stop(self):
        """Stop the keyboard handler."""
        self._running = False

        if self._server:
            self._server.close()

        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread:
            self._thread.join(timeout=1.0)

        logger.info("Keyboard handler stopped")

    def get_paddle_input(self) -> float:
        """Get the current paddle input from keyboard state.

        Returns:
            Float from -1 (down) to 1 (up) based on pressed keys.
        """
        input_value = 0.0

        if self.game_state.is_key_pressed('w'):
            input_value += 1.0
        if self.game_state.is_key_pressed('s'):
            input_value -= 1.0

        return input_value

    @staticmethod
    def get_inject_script() -> str:
        """Get the JavaScript to inject into the browser.

        Returns:
            HTML script tag with keyboard capture code.
        """
        return KEYBOARD_JS
