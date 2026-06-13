# modules/gps_navigator.py — outdoor GPS location announcements  (corrected)
#
# FIXES applied:
#   1. _init_gps() no longer mutates config.GPS_ENABLED as a global side effect.
#      An instance variable self._enabled is used instead.
#   2. Nominatim geocoding now handles HTTP 429 (rate limit) responses gracefully.
#   3. get_location_description() uses self._enabled instead of config.GPS_ENABLED.

import threading
import time
import config

class GPSNavigator:
    """
    Reads GPS coordinates and announces location using reverse geocoding.

    Hardware needed: USB GPS module (e.g. u-blox NEO-6M).
    Set GPS_ENABLED = True and GPS_PORT = "COM3" in config.py (already set for Windows).

    Without hardware: runs in disabled mode (silent).
    """

    def __init__(self):
        self._location      = None
        self._running       = False
        self._thread        = None
        self._last_announce = 0
        # FIX: use instance variable — never mutate config module
        self._enabled       = config.GPS_ENABLED

        if self._enabled:
            self._init_gps()
        else:
            print("[GPSNavigator] GPS disabled. Set GPS_ENABLED=True in config.py to activate.")

    def _init_gps(self):
        """
        FIX: On failure, sets self._enabled = False instead of config.GPS_ENABLED.
        This avoids mutating global state as a side effect of a private method.
        """
        try:
            import serial
            import pynmea2
            self._serial   = serial.Serial(config.GPS_PORT, config.GPS_BAUDRATE, timeout=1)
            self._pynmea2  = pynmea2
            print(f"[GPSNavigator] Connected to GPS on {config.GPS_PORT}")
        except ImportError:
            print("[GPSNavigator] Install: pip install pyserial pynmea2")
            self._enabled = False   # FIX: instance var, not config
        except Exception as e:
            print(f"[GPSNavigator] GPS connection failed: {e}")
            self._enabled = False   # FIX: instance var, not config

    def start(self):
        if not self._enabled:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print("[GPSNavigator] Started.")

    def stop(self):
        self._running = False

    def _read_loop(self):
        """Continuously read GPS NMEA sentences."""
        while self._running:
            try:
                line = self._serial.readline().decode("ascii", errors="replace")
                if line.startswith("$GPRMC") or line.startswith("$GPGGA"):
                    msg = self._pynmea2.parse(line)
                    if hasattr(msg, "latitude") and msg.latitude:
                        self._location = {
                            "lat": msg.latitude,
                            "lon": msg.longitude,
                        }
            except Exception:
                time.sleep(0.5)

    def get_location_description(self):
        """
        Returns a spoken location string or None.
        Uses Nominatim (OpenStreetMap) for reverse geocoding — no API key needed.
        Nominatim ToS: max 1 request/second. MAPS_ANNOUNCE_EVERY=10s gives safe headroom.
        """
        if not self._enabled or not self._location:
            return None

        now = time.time()
        if now - self._last_announce < config.MAPS_ANNOUNCE_EVERY:
            return None

        self._last_announce = now

        try:
            import requests
            lat = self._location["lat"]
            lon = self._location["lon"]

            url = (
                f"https://nominatim.openstreetmap.org/reverse"
                f"?lat={lat}&lon={lon}&format=json"
            )
            resp = requests.get(
                url,
                headers={"User-Agent": "BlindAssistant/1.0"},
                timeout=5
            )

            # FIX: handle Nominatim rate-limit response (HTTP 429)
            if resp.status_code == 429:
                print("[GPSNavigator] Nominatim rate limit hit — backing off.")
                self._last_announce = now + 30   # wait an extra 30s before retrying
                return None

            resp.raise_for_status()
            data = resp.json()

            address = data.get("address", {})
            road    = address.get("road", "")
            suburb  = address.get("suburb", address.get("neighbourhood", ""))
            city    = address.get("city", address.get("town", ""))

            parts = [p for p in [road, suburb, city] if p]
            if parts:
                return f"You are currently on {', '.join(parts)}."
            else:
                return f"Your location: latitude {lat:.4f}, longitude {lon:.4f}."

        except Exception as e:
            print(f"[GPSNavigator] Geocoding failed: {e}")
            return None