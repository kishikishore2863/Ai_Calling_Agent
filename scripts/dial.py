"""Place one outbound call by dispatching the agent into a new room.

  uv run python scripts/dial.py +919876543210 --name "Ravi Kumar"
  uv run python scripts/dial.py --no-phone      # no phone call: agent joins a room you can
                                                # open in the LiveKit playground to talk to it

Phase 1 replaces this with the campaign scheduler + dialer service.
"""

import argparse
import asyncio
import json
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from livekit import api  # noqa: E402

from services.shared.config import get_settings  # noqa: E402

E164 = re.compile(r"^\+[1-9]\d{7,14}$")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phone", nargs="?", help="E.164 number, e.g. +919876543210")
    ap.add_argument("--name", help="contact name")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--no-phone", action="store_true", help="dispatch without dialing")
    args = ap.parse_args()

    if not args.no_phone and not (args.phone and E164.match(args.phone)):
        ap.error("give a phone number in E.164 format (+<country><number>) or use --no-phone")

    s = get_settings()
    room = f"call-{uuid.uuid4().hex[:10]}"
    meta = {"contact_name": args.name}
    if not args.no_phone:
        meta["phone_number"] = args.phone
    if args.profile:
        meta["profile"] = args.profile

    async with api.LiveKitAPI(s.livekit_url, s.livekit_api_key, s.livekit_api_secret) as lk:
        d = await lk.agent_dispatch.create_dispatch(api.CreateAgentDispatchRequest(
            agent_name=s.agent_name, room=room, metadata=json.dumps(meta)))
    print(f"dispatched agent '{s.agent_name}' to room {room} (dispatch {d.id})")
    print(f"result will be written to {s.data_dir / (room + '.json')}")


if __name__ == "__main__":
    asyncio.run(main())
