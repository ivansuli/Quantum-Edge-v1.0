"""
Punto de entrada principal.
Ejecutar: python main.py
"""

import uvicorn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import APP_CONFIG


if __name__ == "__main__":
    uvicorn.run(
        "api.server:app",
        host=APP_CONFIG.host,
        port=APP_CONFIG.port,
        reload=APP_CONFIG.debug,
        log_level="info",
    )
