"""
Server startup script for the Customer Churn Prediction API.

This script provides a convenient way to start the FastAPI server with
proper configuration and logging.
"""

import uvicorn
import sys
from pathlib import Path
import argparse
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logging_setup import get_notebook_logger

logger = get_notebook_logger(__name__)

def main():
    """Main function to start the API server."""
    parser = argparse.ArgumentParser(description="Start the Customer Churn Prediction API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"], 
                       help="Log level (default: info)")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (default: 1)")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Customer Churn Prediction API server...")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Reload: {args.reload}")
    logger.info(f"Log level: {args.log_level}")
    logger.info(f"Workers: {args.workers}")
    
    try:
        # Start the server
        uvicorn.run(
            "api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            workers=args.workers if not args.reload else 1,  # Can't use multiple workers with reload
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()