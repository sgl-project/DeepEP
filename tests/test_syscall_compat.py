import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestSyscallCompat(unittest.TestCase):
    def _compile(self, syscall_header: str, assertions: str, arch: str = None):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            sys_dir = tmp_path / 'sys'
            sys_dir.mkdir()
            (sys_dir / 'syscall.h').write_text(syscall_header)
            source = tmp_path / 'test.cpp'
            source.write_text(
                '#include <deep_ep/common/syscall.cuh>\n'
                f'{assertions}\n'
                'int main() { return 0; }\n'
            )
            command = [
                'c++',
                '-std=c++17',
                '-fsyntax-only',
                '-D__linux__',
                '-U__x86_64__',
                '-U__aarch64__',
            ]
            if arch is not None:
                command.append(f'-D{arch}')
            command.extend([
                '-I',
                str(tmp_path),
                '-I',
                str(REPO_ROOT / 'deep_ep' / 'include'),
                str(source),
            ])
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_defines_pidfd_syscalls_for_manylinux_headers(self):
        for arch in ('__x86_64__', '__aarch64__'):
            with self.subTest(arch=arch):
                self._compile(
                    '',
                    'static_assert(SYS_pidfd_open == 434);\n'
                    'static_assert(SYS_pidfd_getfd == 438);',
                    arch,
                )

    def test_prefers_kernel_syscall_definitions(self):
        self._compile(
            '#define __NR_pidfd_open 1234\n#define __NR_pidfd_getfd 5678\n',
            'static_assert(SYS_pidfd_open == 1234);\n'
            'static_assert(SYS_pidfd_getfd == 5678);',
        )

    def test_preserves_libc_syscall_definitions(self):
        self._compile(
            '#define SYS_pidfd_open 1234\n#define SYS_pidfd_getfd 5678\n',
            'static_assert(SYS_pidfd_open == 1234);\n'
            'static_assert(SYS_pidfd_getfd == 5678);',
        )


if __name__ == '__main__':
    unittest.main()
