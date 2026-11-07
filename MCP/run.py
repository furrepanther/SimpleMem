#!/usr/bin/env python3
"""
SimpleMem MCP Server - Runner Script

Starts the HTTP server which provides:
- Web UI for configuration (/)
- REST API for memory operations (/api/*)
- MCP over Streamable HTTP for Claude Desktop (/mcp)
- Legacy SSE compatibility remains available at (/mcp/sse)
"""

import argparse


# This applies only after an explicit shutdown signal.  It is not a watchdog
# or a runtime deadline: it gives long-lived MCP streams time to drain during
# a deliberate maintenance quiesce, then lets Uvicorn cancel a stream that
# would otherwise hold the process forever after it has stopped accepting new
# work.  The Phase-0 quiescer observes for 60 seconds, reserving 45 seconds
# for application cleanup after the stream-draining window.  It never
# escalates to SIGKILL.
MAINTENANCE_GRACEFUL_SHUTDOWN_S = 15


def main():
    parser = argparse.ArgumentParser(
        description="SimpleMem MCP Server - Multi-tenant Memory Service for LLM Agents"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("SimpleMem MCP must bind to a loopback host in the production runtime")

    print("=" * 60)
    print("  SimpleMem MCP Server")
    print("  Multi-tenant Memory Service for LLM Agents")
    print("=" * 60)
    print()
    print(f"  Web UI:     http://localhost:{args.port}/")
    print(f"  REST API:   http://localhost:{args.port}/api/")
    print(f"  MCP:        http://localhost:{args.port}/mcp")
    print(f"  MCP (legacy SSE): http://localhost:{args.port}/mcp/sse")
    print()
    print("-" * 60)

    import uvicorn
    uvicorn.run(
        "server.http_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        timeout_graceful_shutdown=MAINTENANCE_GRACEFUL_SHUTDOWN_S,
    )


if __name__ == "__main__":
    main()
