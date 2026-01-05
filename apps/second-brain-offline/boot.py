import os
import threading
import runpy
from http.server import HTTPServer, BaseHTTPRequestHandler
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


def run_pipeline_once():
    try:
        logger.info("🚀 Starting compute_rag_vector_index pipeline on boot")

        runpy.run_module(
            "pipelines.compute_rag_vector_index",
            run_name="__main__",
        )

        logger.info("✅ compute_rag_vector_index pipeline finished successfully")

    except Exception:
        logger.exception("❌ Pipeline execution failed")


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()


if __name__ == "__main__":
    # 1️⃣ START HEALTH SERVER FIRST (CRITICAL)
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🩺 Binding health server on port {port}")

    server = HTTPServer(("0.0.0.0", port), HealthHandler)

    # 2️⃣ RUN PIPELINE IN BACKGROUND
    threading.Thread(
        target=run_pipeline_once,
        daemon=True,
    ).start()

    # 3️⃣ KEEP PROCESS ALIVE
    server.serve_forever()
