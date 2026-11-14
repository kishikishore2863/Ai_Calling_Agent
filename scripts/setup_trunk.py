"""Register your SIP provider as an outbound trunk in (self-hosted) LiveKit SIP.

Set these in .env first (values come from your SIP provider's dashboard):
  SIP_TRUNK_ADDRESS=sip.telnyx.com          # provider's SIP domain / termination URI
  SIP_TRUNK_NUMBER=+91XXXXXXXXXX            # your caller ID number on that trunk
  SIP_TRUNK_USERNAME=...                    # credential auth (or IP-allowlist your server)
  SIP_TRUNK_PASSWORD=...

  uv run python scripts/setup_trunk.py          # create trunk, print SIP_OUTBOUND_TRUNK_ID
  uv run python scripts/setup_trunk.py --list   # show existing trunks
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402
from livekit import api  # noqa: E402

from services.shared.config import get_settings  # noqa: E402


async def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    s = get_settings()

    async with api.LiveKitAPI(s.livekit_url, s.livekit_api_key, s.livekit_api_secret) as lk:
        if args.list:
            res = await lk.sip.list_outbound_trunk(api.ListSIPOutboundTrunkRequest())
            for t in res.items:
                print(t.sip_trunk_id, t.name, t.address, list(t.numbers))
            return

        missing = [k for k in ("SIP_TRUNK_ADDRESS", "SIP_TRUNK_NUMBER") if not os.getenv(k)]
        if missing:
            sys.exit(f"missing in .env: {', '.join(missing)}")
        trunk = await lk.sip.create_outbound_trunk(api.CreateSIPOutboundTrunkRequest(
            trunk=api.SIPOutboundTrunkInfo(
                name="primary-outbound",
                address=os.environ["SIP_TRUNK_ADDRESS"],
                numbers=[os.environ["SIP_TRUNK_NUMBER"]],
                auth_username=os.getenv("SIP_TRUNK_USERNAME", ""),
                auth_password=os.getenv("SIP_TRUNK_PASSWORD", ""),
            )))
    print(f"created trunk. Add to .env:\nSIP_OUTBOUND_TRUNK_ID={trunk.sip_trunk_id}")


if __name__ == "__main__":
    asyncio.run(main())
