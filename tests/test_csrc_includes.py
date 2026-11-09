"""Every quoted #include in the packaged csrc must resolve inside the shipped tree (JIT builds fail at compile time otherwise)."""
import re, unittest
from pathlib import Path

CSRC = Path(__file__).resolve().parents[1] / 'src' / 'atdev_serve' / 'csrc'
# extra include dir (-I, relative to csrc) the JIT builders add for sources under a top-level csrc directory
EXTRA = {'marlin_plus_w3': ['marlin_plus', 'marlin_plus_w3'], 'marlin_plus_a4': ['marlin_plus', 'marlin_plus_a4']}
# written by the *_build.py helpers into the torch-extension cache at build time
GENERATED = re.compile(r'(mp_selector|w3_selector|mp_chain_selector|a4_selector|a4_reduce_selector|a4ws_selector)\.h$')


class CsrcIncludes(unittest.TestCase):
    def test_quoted_includes_resolve(self):
        files = [p for p in CSRC.rglob('*') if p.suffix in ('.cu', '.cuh', '.h', '.hpp', '.inc')]
        self.assertGreater(len(files), 30)
        missing = []
        for p in files:
            for n, line in enumerate(p.read_text(errors='ignore').splitlines(), 1):
                m = re.match(r'\s*#\s*include\s+"([^"]+)"', line)
                if not m or GENERATED.search(m.group(1)):
                    continue
                t = m.group(1)
                top = p.relative_to(CSRC).parts[0]
                dirs = [p.parent, CSRC, CSRC / top] + [CSRC / r for r in EXTRA.get(top, [])]
                if not any((d / t).exists() for d in dirs):
                    missing.append(f'{p.relative_to(CSRC)}:{n}: {t}')
        self.assertEqual(missing, [])

    def test_check_detects_missing(self):
        # guard against the scan silently matching nothing: the Marlin dtype headers include core/scalar_type.hpp
        text = (CSRC / 'marlin_c1' / 'marlin_dtypes.cuh').read_text()
        self.assertIn('#include "core/scalar_type.hpp"', text)
        self.assertTrue((CSRC / 'marlin_c1' / 'core' / 'scalar_type.hpp').exists())
        self.assertTrue((CSRC / 'marlin_plus' / 'core' / 'scalar_type.hpp').exists())


if __name__ == '__main__':
    unittest.main()
