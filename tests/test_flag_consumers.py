"""Every ATDEV_* switch a launch profile sets must be read by some module of the package (a flag nobody reads is
silently ignored by the plugin, which would make a profile quietly slower or different), and every module a
profile enables through ATDEV_OPT or a convenience flag must be registered."""
import re
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
SRC = ROOT / 'src' / 'atdev_serve'
# set by the scripts for the launch environment itself, not read by the package
NOT_READ_BY_PACKAGE = {'ATDEV_OPT'}


def package_text():
    return '\n'.join(p.read_text(errors='ignore') for p in SRC.rglob('*')
                     if p.suffix in {'.py', '.cu', '.cuh', '.h', '.inc'} and 'gen' not in p.parts[-2][:3])


class Consumers(unittest.TestCase):
    def test_every_flag_exported_by_the_launch_scripts_is_read_by_the_package(self):
        text = package_text()
        for script in ('serve_c1.sh', 'serve_c32.sh', 'serve_c1_accuracy.sh'):
            body = (ROOT / script).read_text()
            body = '\n'.join(l for l in body.splitlines() if not l.lstrip().startswith('#') or 'export' in l)
            flags = set(re.findall(r'\b(ATDEV_[A-Z0-9_]+)=', body)) - NOT_READ_BY_PACKAGE
            missing = sorted(f for f in flags if not re.search(r'\b' + f + r'\b', text))
            self.assertEqual(missing, [], f'{script}: flags nobody reads')

    def test_opt_names_and_convenience_flags_are_registered(self):
        from atdev_serve import register
        for script in ('serve_c1.sh', 'serve_c32.sh'):
            for names in re.findall(r'ATDEV_OPT=([a-z0-9_,]+)', (ROOT / script).read_text()):
                for n in names.split(','):
                    self.assertIn(n, register.OPTIMIZATIONS, f'{script}: {n}')
        for mod in ('pf_fa2', 'pf_fi'):
            self.assertIn(mod, register.OPTIMIZATIONS)

    def test_register_installs_every_frontend_hook(self):
        body = (SRC / 'register.py').read_text()
        for flag, module in (('ATDEV_SWA_FINE', 'swa_fine'), ('ATDEV_TOK_CACHE', 'tok_cache'), ('ATDEV_GDN_SMALL', 'gdn_small'),
                             ('ATDEV_EOR_CKPT', 'eor_ckpt'), ('ATDEV_DFT_APC', 'GDN_SS_ALIGN')):
            self.assertIn(flag, body)
            self.assertIn(module, body)
            if module != 'GDN_SS_ALIGN':
                self.assertTrue((SRC / f'{module}.py').exists(), module)

    def test_default_draft_vocab_order_is_the_packaged_file(self):
        from atdev_serve import draft_vocab
        self.assertTrue(Path(draft_vocab._DEFAULT_FILE).resolve().is_relative_to(SRC.resolve()))
        self.assertTrue(Path(draft_vocab._DEFAULT_FILE).exists())


if __name__ == '__main__':
    unittest.main()
