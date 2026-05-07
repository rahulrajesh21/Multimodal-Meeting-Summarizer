import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run():
    token = os.environ.get("ATLASSIAN_API_TOKEN", "")
    server_params = StdioServerParameters(
        command="npx",
        args=[
            "-y", "mcp-remote",
            "https://mcp.atlassian.com/v1/mcp",
            "--header", f"Authorization: Bearer {token}",
        ],
        env={**os.environ}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List tools
            tools_resp = await session.list_tools()
            for t in tools_resp.tools:
                print("---")
                print("Name:", t.name)
                print("Desc:", t.description)
                print("Schema:", t.inputSchema)

asyncio.run(run())
