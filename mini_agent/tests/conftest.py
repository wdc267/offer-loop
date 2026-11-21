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
    """提供同步执行协程的辅助函数，避免引入 pytest-asyncio。

    返回值：
        一个接收协程并返回其结果的 `_run` 函数。
    """

    def _run(coro):
        """在新事件循环中执行协程并返回结果。

        参数：
            coro: 待执行的协程对象。

        返回值：
            协程的返回值。
        """
        return asyncio.run(coro)

    return _run
