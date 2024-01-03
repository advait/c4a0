#!/usr/bin/env python

import asyncio
import logging

from training import train_gen


async def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    await train_gen()


if __name__ == "__main__":
    asyncio.run(main())
