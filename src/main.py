#!/usr/bin/env python

import asyncio
import logging

from training import train


async def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    while True:
        await train()


if __name__ == "__main__":
    asyncio.run(main())
