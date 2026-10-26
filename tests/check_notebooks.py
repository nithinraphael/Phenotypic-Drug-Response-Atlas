import asyncio

from marimo._ast.load import get_notebook_status
from marimo._lint.rule_engine import RuleEngine
from marimo._lint.rules import DEFAULT_RULE_CODES


async def main():
    for path in ('notebook.py','dev_notebook.py'):
        loaded=get_notebook_status(path)
        assert loaded.status=='valid',path
        for code,rule in DEFAULT_RULE_CODES.items():
            diagnostics=await RuleEngine([rule()]).check_notebook(loaded.notebook,loaded.contents)
            assert not diagnostics,(path,code,diagnostics)
        print(f'{path}: all {len(DEFAULT_RULE_CODES)} default marimo rules passed')


if __name__=='__main__':
    asyncio.run(main())
