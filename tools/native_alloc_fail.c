/* Linux/glibc-only test helper. Fail one allocation made directly by the
 * HOTPANTS extension; leave Python/NumPy and cleanup allocations alone.
 * Compile as a preload library and control it with ctypes from memory_probe.py.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <string.h>
extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
static long countdown = -1;
static int injected = 0;
static __thread int checking = 0;
void hp_fail_after(long n)
{
    countdown = n;
    injected = 0;
}
int hp_did_fail(void) { return injected; }
static int should_fail(void *caller)
{
    if (countdown < 0 || checking)
        return 0;
    checking = 1;
    Dl_info info;
    int matches = dladdr(caller, &info) && info.dli_fname && strstr(info.dli_fname, "hotpants_ext");
    checking = 0;
    if (matches && countdown-- == 0)
    {
        injected = 1;
        countdown = -1;
        return 1;
    }
    return 0;
}
void *malloc(size_t n)
{
    return should_fail(__builtin_return_address(0)) ? NULL : __libc_malloc(n);
}
void *calloc(size_t n, size_t size)
{
    return should_fail(__builtin_return_address(0)) ? NULL : __libc_calloc(n, size);
}
void *realloc(void *p, size_t n)
{
    return should_fail(__builtin_return_address(0)) ? NULL : __libc_realloc(p, n);
}
