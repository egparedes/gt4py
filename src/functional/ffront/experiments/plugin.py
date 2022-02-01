from typing import Callable, Optional

import mypy
import mypy.plugin as my_plug
import mypy.types as my_types


class CustomPlugin(my_plug.Plugin):
    def get_type_analyze_hook(
        self, fullname: str
    ) -> Optional[Callable[[my_plug.AnalyzeTypeContext], my_types.Type]]:
        """Customize behaviour of the type analyzer for given full names.
        This method is called during the semantic analysis pass whenever mypy sees an
        unbound type. For example, while analysing this code:
            from lib import Special, Other
            var: Special
            def func(x: Other[int]) -> None:
                ...
        this method will be called with 'lib.Special', and then with 'lib.Other'.
        The callback returned by plugin must return an analyzed type,
        i.e. an instance of `mypy.types.Type`.
        """
        print(f"\n\n CustomPlugin.get_type_analyze_hook({fullname})\n\n")
        return None


def plugin(version: str) -> my_plug.Plugin:
    # ignore version argument if the plugin works with all mypy versions.
    # print(f"\n\n MYPY: loading plugin!! ({version})\n\n")
    return CustomPlugin
