# SU(2) Fourier Transform: Direct (O(N^6)) and Fast (O(N^4))
# arxiv 2605.23923 -- ground truth in paper.tex, summarised in notes/.
# Literate companion: ALGORITHM.md.

CC        ?= cc
CSTD      ?= -std=c11
WARN      ?= -Wall -Wextra -Wpedantic -Wshadow -Wstrict-prototypes
OPT       ?= -O2 -g
INCLUDES  := -Iinclude -I/usr/include
DEFINES   := -D_GNU_SOURCE
LIBS      := -lfftw3 -lflint -lm
CFLAGS    := $(CSTD) $(WARN) $(OPT) $(INCLUDES) $(DEFINES) -fPIC

SRC_DIR   := src
TEST_DIR  := tests
BENCH_DIR := bench
BUILD_DIR := build

LIB_SRCS  := $(wildcard $(SRC_DIR)/*.c)
LIB_OBJS  := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(LIB_SRCS))

TEST_SRCS := $(wildcard $(TEST_DIR)/*.c)
TEST_BINS := $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/%,$(TEST_SRCS))

BENCH_SRCS := $(wildcard $(BENCH_DIR)/*.c)
BENCH_BINS := $(patsubst $(BENCH_DIR)/%.c,$(BUILD_DIR)/%,$(BENCH_SRCS))

.PHONY: all test bench clean lib

all: $(LIB_OBJS) $(TEST_BINS) $(BENCH_BINS)

lib: $(BUILD_DIR)/libsu2.so

$(BUILD_DIR)/libsu2.so: $(LIB_OBJS) | $(BUILD_DIR)
	$(CC) -shared $(LIB_OBJS) $(LIBS) -o $@

# Compile library objects
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Build test binaries (each test links the full library)
$(BUILD_DIR)/%: $(TEST_DIR)/%.c $(LIB_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(LIB_OBJS) $(LIBS) -o $@

# Build benchmark binaries
$(BUILD_DIR)/%: $(BENCH_DIR)/%.c $(LIB_OBJS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $< $(LIB_OBJS) $(LIBS) -o $@

$(BUILD_DIR):
	mkdir -p $@

# Run every test binary in sequence; abort on first failure.
test: $(TEST_BINS)
	@set -e; for t in $(TEST_BINS); do \
		echo "=== $$t ==="; \
		$$t || { echo "FAIL: $$t"; exit 1; }; \
	done; \
	echo "=== ALL TESTS PASSED ==="

bench: $(BENCH_BINS)
	@for b in $(BENCH_BINS); do echo "=== $$b ==="; $$b; done

clean:
	rm -rf $(BUILD_DIR)
