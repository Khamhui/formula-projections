"""
Desktop app — native macOS window for the F1 dashboard.

Uses pywebview (WebKit) to display the Flask dashboard in a proper window
instead of a browser tab. Shows in Dock, Cmd+Tab, supports fullscreen.

Usage:
    python -m src.desktop              # Default size
    python -m src.desktop --fullscreen # Start fullscreen
"""

from __future__ import annotations

import sys
import threading

import webview

from src.app import app

PORT = 5050


def _ensure_foreground():
    """Register this process as a foreground GUI app with macOS.

    When launched from a .app bundle shell script, the process doesn't get
    a proper WindowServer connection. Calling NSApp.setActivationPolicy_
    before pywebview starts fixes this.
    """
    if sys.platform != "darwin":
        return
    try:
        from AppKit import NSApplication, NSApplicationActivationPolicyRegular
        ns_app = NSApplication.sharedApplication()
        ns_app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
        ns_app.activateIgnoringOtherApps_(True)
    except ImportError:
        pass


def _run_flask():
    """Run Flask in a background thread (no browser, no reloader)."""
    app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False)


def main(fullscreen: bool = False):
    _ensure_foreground()

    window = webview.create_window(
        title="F1 Predictions",
        url=f"http://127.0.0.1:{PORT}",
        width=1440,
        height=900,
        min_size=(1024, 600),
        text_select=True,
    )

    # Start Flask before the webview event loop blocks
    server = threading.Thread(target=_run_flask, daemon=True)
    server.start()

    webview.start(
        gui="cocoa",
        debug=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="F1 Desktop App")
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()

    main(fullscreen=args.fullscreen)
