/* test_framework.h -- minimal red/green test harness.
 * No dependencies beyond libc and complex.h. */
#ifndef TEST_FRAMEWORK_H
#define TEST_FRAMEWORK_H

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int g_test_failures = 0;
static int g_test_total    = 0;

#define TEST_BEGIN(name) do {                                                 \
    g_test_total++;                                                           \
    fprintf(stderr, "  [test] %s ... ", (name));                              \
} while (0)

#define TEST_PASS() do { fprintf(stderr, "OK\n"); } while (0)

#define TEST_FAIL(fmt, ...) do {                                              \
    g_test_failures++;                                                        \
    fprintf(stderr, "FAIL\n    " fmt "\n", ##__VA_ARGS__);                    \
    return;                                                                   \
} while (0)

#define ASSERT_TRUE(cond) do {                                                \
    if (!(cond)) TEST_FAIL("ASSERT_TRUE(%s) at %s:%d", #cond, __FILE__, __LINE__); \
} while (0)

#define ASSERT_NEAR(a, b, tol) do {                                           \
    double _a = (a), _b = (b), _t = (tol);                                    \
    if (!(fabs(_a - _b) <= _t)) TEST_FAIL(                                    \
        "ASSERT_NEAR(%s=%.17g, %s=%.17g, tol=%g) at %s:%d -- |diff|=%g",     \
        #a, _a, #b, _b, _t, __FILE__, __LINE__, fabs(_a - _b));               \
} while (0)

#define ASSERT_CNEAR(a, b, tol) do {                                          \
    double _Complex _a = (a), _b = (b);                                       \
    double _t = (tol);                                                        \
    double _d = cabs(_a - _b);                                                \
    if (!(_d <= _t)) TEST_FAIL(                                               \
        "ASSERT_CNEAR(%s=(%.6g%+.6gi), %s=(%.6g%+.6gi), tol=%g) at %s:%d -- |diff|=%g", \
        #a, creal(_a), cimag(_a), #b, creal(_b), cimag(_b), _t,               \
        __FILE__, __LINE__, _d);                                              \
} while (0)

#define TEST_REPORT_AND_EXIT() do {                                           \
    fprintf(stderr, "  ---- %d/%d tests passed ----\n",                       \
            g_test_total - g_test_failures, g_test_total);                    \
    return g_test_failures == 0 ? 0 : 1;                                      \
} while (0)

#define RUN(fn) do { TEST_BEGIN(#fn); fn(); TEST_PASS(); } while (0)

#endif /* TEST_FRAMEWORK_H */
