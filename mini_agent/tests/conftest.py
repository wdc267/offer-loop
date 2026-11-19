"""测试公共配置。

故意不引入 pytest-asyncio：
- FakeProvider 没有跨事件循环持有资源；
- 每个测试用 asyncio.run() 跑完就结束，足够且少一个依赖。
"""
from __future__ import annotations

import asyncio

import pytest


@pytest.fixture
def run_async():
    def _run(coro):
        return asyncio.run(coro)

    return _run
