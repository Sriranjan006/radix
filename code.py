
"""
Corrected, reproducible benchmark for:

R1/R2/R3 = Conventional LSD Radix Sort (1/2/3-bit digits)
C1/C2/C3 = Bit-Length-Compartment Radix Sort (1/2/3-bit digits)
QS       = Iterative three-way Quick Sort
MS       = Bottom-up Merge Sort
PS       = Python sorted()

Benchmark modes
---------------
1. sorting_only:
   - A fresh input copy is created outside the timed interval.
   - Sorting and algorithm-owned workspace are timed.
   - All custom sorting algorithms operate in place.
   - Python sorted() naturally creates and returns a new list.

2. end_to_end:
   - A fresh input copy is created inside the timed interval.
   - Copying and sorting are both timed.
   - Python sorted() additionally creates its normal output list.

Memory
------
- Memory is measured in separate runs.
- tracemalloc is never active during timing.
- Memory measurements include the caller-side input copy and
  algorithm-owned temporary workspace/output.

Run
---
python corrected_benchmark.py --quick
python corrected_benchmark.py
python corrected_benchmark.py --mode end_to_end
python corrected_benchmark.py --sizes 10000 50000 100000 --repeats 20
"""

import argparse
import csv
import random
import statistics
import time
import tracemalloc
from pathlib import Path


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

ARRAY_SIZES = [
    10_000,
    50_000,
    100_000,
    500_000,
    1_000_000
]

VALUE_RANGES = [
    (0, 255),
    (0, 100_000),
    (0, 1_000_000),
    (100_000, 1_000_000)
]

PATTERNS = [
    "random",
    "sorted",
    "reverse",
    "duplicates",
    "skewed",
    "narrow_range"
]

DEFAULT_REPEATS = 20
DEFAULT_MEMORY_REPEATS = 3

RANDOM_SEED = 12345
ORDER_SEED = 12346


# ============================================================
# 2. CONVENTIONAL LSD RADIX SORT
# ============================================================

def radix_sort(values, digit_bits):

    if digit_bits <= 0:
        raise ValueError("digit_bits must be positive")

    if not values:
        return values

    if any(x < 0 for x in values):
        raise ValueError(
            "Only non-negative integers are supported"
        )

    # Work directly on the supplied list.
    result = values

    maximum = max(result)

    if maximum == 0:
        return result

    output = [0] * len(result)

    base = 1 << digit_bits
    mask = base - 1

    shift = 0

    while (maximum >> shift) > 0:

        counts = [0] * base

        # Count digit occurrences.
        for value in result:
            counts[(value >> shift) & mask] += 1

        # Convert counts into starting positions.
        total = 0

        for digit in range(base):

            count = counts[digit]
            counts[digit] = total
            total += count

        # Stable distribution.
        for value in result:

            digit = (value >> shift) & mask

            output[counts[digit]] = value
            counts[digit] += 1

        # Swap source and destination.
        result, output = output, result

        shift += digit_bits

    # Ensure the final sorted result is in the original list.
    if result is not values:
        values[:] = result

    return values


# ============================================================
# 3. BIT-LENGTH-COMPARTMENT RADIX SORT
# ============================================================

def compartment_radix_sort(values, digit_bits):

    if digit_bits <= 0:
        raise ValueError("digit_bits must be positive")

    if len(values) <= 1:

        if values and values[0] < 0:
            raise ValueError(
                "Only non-negative integers are supported"
            )

        return

    compartments = {}
    zero_count = 0

    # Group numbers according to their bit lengths.
    for value in values:

        if value < 0:
            raise ValueError(
                "Only non-negative integers are supported"
            )

        if value == 0:

            zero_count += 1

        else:

            bits = value.bit_length()

            compartments.setdefault(
                bits, []
            ).append(value)

    # Sort each compartment independently.
    for bits, component in compartments.items():

        if (
            len(component) <= 1
            or all(x == component[0] for x in component)
        ):
            continue

        output = [0] * len(component)

        # The leading 1 bit is shared by all numbers
        # in this compartment, so it is not processed.
        for shift in range(0, bits - 1, digit_bits):

            width = min(
                digit_bits,
                bits - 1 - shift
            )

            base = 1 << width
            mask = base - 1

            counts = [0] * base

            # Count digit occurrences.
            for value in component:

                digit = (value >> shift) & mask
                counts[digit] += 1

            # Calculate starting positions.
            total = 0

            for digit in range(base):

                count = counts[digit]
                counts[digit] = total
                total += count

            # Stable distribution.
            for value in component:

                digit = (value >> shift) & mask

                output[counts[digit]] = value
                counts[digit] += 1

            component, output = output, component

        compartments[bits] = component

    # Reconstruct the original list in ascending order.
    values[:] = [0] * zero_count + [
        value
        for bits in sorted(compartments)
        for value in compartments[bits]
    ]


# ============================================================
# 4. ITERATIVE THREE-WAY QUICK SORT
# ============================================================

def quick_sort(values):

    # Operates directly on the supplied list.
    arr = values

    if len(arr) <= 1:
        return arr

    stack = [(0, len(arr) - 1)]

    while stack:

        low, high = stack.pop()

        if low >= high:
            continue

        mid = (low + high) // 2

        candidates = sorted((
            arr[low],
            arr[mid],
            arr[high]
        ))

        pivot = candidates[1]

        lt = i = low
        gt = high

        # Three-way partitioning.
        while i <= gt:

            if arr[i] < pivot:

                arr[lt], arr[i] = arr[i], arr[lt]

                lt += 1
                i += 1

            elif arr[i] > pivot:

                arr[i], arr[gt] = arr[gt], arr[i]

                gt -= 1

            else:

                i += 1

        left_size = lt - low
        right_size = high - gt

        # Push the larger partition first,
        # so the smaller partition is processed next.
        if left_size < right_size:

            if gt + 1 < high:
                stack.append((gt + 1, high))

            if low < lt - 1:
                stack.append((low, lt - 1))

        else:

            if low < lt - 1:
                stack.append((low, lt - 1))

            if gt + 1 < high:
                stack.append((gt + 1, high))

    return arr


# ============================================================
# 5. ITERATIVE BOTTOM-UP MERGE SORT
# ============================================================

def merge_sort(values):

    n = len(values)

    if n <= 1:
        return values

    # Use the supplied list as the initial source.
    source = values

    target = [0] * n

    width = 1

    while width < n:

        for left in range(0, n, 2 * width):

            mid = min(left + width, n)
            right = min(left + 2 * width, n)

            i = left
            j = mid
            k = left

            # Merge two sorted sections.
            while i < mid and j < right:

                if source[i] <= source[j]:

                    target[k] = source[i]
                    i += 1

                else:

                    target[k] = source[j]
                    j += 1

                k += 1

            # Copy remaining left elements.
            while i < mid:

                target[k] = source[i]

                i += 1
                k += 1

            # Copy remaining right elements.
            while j < right:

                target[k] = source[j]

                j += 1
                k += 1

        source, target = target, source

        width *= 2

    # Ensure the final output is in the original list.
    if source is not values:
        values[:] = source

    return values


# ============================================================
# 6. PYTHON BUILT-IN SORTING
# ============================================================

def python_sorted(values):

    # Python sorted() returns a new list.
    return sorted(values)


# ============================================================
# 7. ALGORITHM WRAPPERS
# ============================================================

def radix_1bit(values):
    return radix_sort(values, 1)


def radix_2bit(values):
    return radix_sort(values, 2)


def radix_3bit(values):
    return radix_sort(values, 3)


def compartment_1bit(values):
    compartment_radix_sort(values, 1)


def compartment_2bit(values):
    compartment_radix_sort(values, 2)


def compartment_3bit(values):
    compartment_radix_sort(values, 3)


# ============================================================
# 8. ALGORITHM REGISTRY
# ============================================================

# Format:
# (algorithm name, function, mutates_input)

ALGORITHMS = [

    ("Radix 1-bit", radix_1bit, True),
    ("Radix 2-bit", radix_2bit, True),
    ("Radix 3-bit", radix_3bit, True),

    ("Compartment 1-bit", compartment_1bit, True),
    ("Compartment 2-bit", compartment_2bit, True),
    ("Compartment 3-bit", compartment_3bit, True),

    ("Quick Sort", quick_sort, True),
    ("Merge Sort", merge_sort, True),

    ("Python sorted", python_sorted, False)
]


# ============================================================
# 9. INPUT GENERATION
# ============================================================

def make_input(size, low, high, pattern, rng):

    if size < 0:
        raise ValueError("size must be non-negative")

    if low < 0 or high < low:
        raise ValueError(
            "Value bounds must satisfy 0 <= low <= high"
        )

    if pattern not in PATTERNS:
        raise ValueError(
            f"Unknown pattern: {pattern}"
        )

    # Duplicate-heavy input.
    if pattern == "duplicates":

        upper = min(high, low + 10)

        return [
            rng.randint(low, upper)
            for _ in range(size)
        ]

    # Narrow-range input.
    if pattern == "narrow_range":

        upper = min(
            high,
            low + max(10, (high - low) // 100)
        )

        return [
            rng.randint(low, upper)
            for _ in range(size)
        ]

    # Skewed input.
    if pattern == "skewed":

        clustered_count = int(size * 0.95)

        cluster_high = min(
            high,
            low + max(10, (high - low) // 20)
        )

        values = [
            rng.randint(low, cluster_high)
            for _ in range(clustered_count)
        ]

        values.extend(
            rng.randint(low, high)
            for _ in range(size - clustered_count)
        )

        rng.shuffle(values)

        return values

    # Random input.
    values = [
        rng.randint(low, high)
        for _ in range(size)
    ]

    if pattern == "random":
        return values

    if pattern == "sorted":
        return sorted(values)

    if pattern == "reverse":
        return sorted(values, reverse=True)

    raise ValueError(f"Unknown pattern: {pattern}")


# ============================================================
# 10. CORRECTNESS TESTING
# ============================================================

def verify_algorithms():

    cases = [

        [],
        [0],
        [5],
        [0, 0, 0],
        [1, 1, 1],

        [3, 2, 1, 0],

        [255, 256, 257, 1023, 1024],

        [8, 4, 2, 1, 16, 32, 0],

        [2**31 - 1, 0, 2**31, 1, 2**16]
    ]

    rng = random.Random(999)

    # Generate additional random test cases.
    for _ in range(200):

        cases.append([
            rng.randint(0, 1_000_000)
            for _ in range(rng.randint(0, 500))
        ])

    for case_no, original in enumerate(cases, 1):

        expected = sorted(original)

        for name, func, mutates in ALGORITHMS:

            values = original[:]

            result = func(values)

            if mutates:
                result = values

            if result != expected:

                raise AssertionError(
                    f"{name} failed case {case_no}: "
                    f"expected {expected}, got {result}"
                )

            if mutates and values != expected:

                raise AssertionError(
                    f"{name} did not sort its input in place"
                )

    # Negative-value tests for radix algorithms.
    for func in (

        radix_1bit,
        radix_2bit,
        radix_3bit,

        compartment_1bit,
        compartment_2bit,
        compartment_3bit

    ):

        try:

            func([-1, 0, 1])

        except ValueError:

            pass

        else:

            raise AssertionError(
                f"{func.__name__} must reject negative values"
            )

    print("Correctness tests passed for all nine algorithms.")


def verify_edge_cases():

    """Check in-place behavior and invalid arguments."""

    for digit_bits in (1, 2, 3):

        for func in (
            radix_sort,
            compartment_radix_sort
        ):

            for original in (
                [],
                [0],
                [7],
                [0, 0, 0],
                [9, 1, 9, 0, 1]
            ):

                values = original[:]

                result = func(values, digit_bits)

                if func is compartment_radix_sort:

                    if (
                        result is not None
                        or values != sorted(original)
                    ):

                        raise AssertionError(
                            f"{func.__name__} edge case failed"
                        )

                elif (
                    result is not values
                    or values != sorted(original)
                ):

                    raise AssertionError(
                        f"{func.__name__} edge case failed"
                    )

            # Invalid digit-width test.
            try:

                func([1], 0)

            except ValueError:

                pass

            else:

                raise AssertionError(
                    f"{func.__name__} must reject digit_bits <= 0"
                )

    print("Edge-case and mutability checks passed.")


# ============================================================
# 11. TIMING BENCHMARK
# ============================================================

def benchmark_timings(
    original,
    expected,
    repeats,
    mode,
    order_rng
):

    rows = []

    # --------------------------------------------------------
    # Warm-up runs
    # --------------------------------------------------------

    warmup_order = ALGORITHMS[:]

    order_rng.shuffle(warmup_order)

    for name, func, mutates in warmup_order:

        values = original[:]

        result = func(values)

        if mutates:
            result = values

        if result != expected:

            raise RuntimeError(
                f"{name} failed during warm-up"
            )

    # --------------------------------------------------------
    # Actual timing runs
    # --------------------------------------------------------

    for repetition in range(1, repeats + 1):

        order = ALGORITHMS[:]

        order_rng.shuffle(order)

        for name, func, mutates in order:

            if mode == "sorting_only":

                # Input copying is outside the timed interval.
                values = original[:]

                start = time.perf_counter_ns()

                result = func(values)

            else:

                # Input copying and sorting are both timed.
                start = time.perf_counter_ns()

                values = original[:]

                result = func(values)

            elapsed_ns = (
                time.perf_counter_ns() - start
            )

            if mutates:
                result = values

            # Verify every measured result.
            if result != expected:

                raise RuntimeError(
                    f"{name} returned an incorrect result"
                )

            rows.append({

                "repetition": repetition,

                "algorithm": name,

                "time_ms": elapsed_ns / 1_000_000,

                "timing_mode": mode

            })

    return rows


# ============================================================
# 12. MEMORY BENCHMARK
# ============================================================

def benchmark_memory(
    original,
    expected,
    memory_repeats
):

    rows = []

    for name, func, mutates in ALGORITHMS:

        for repetition in range(1, memory_repeats + 1):

            tracemalloc.stop()

            tracemalloc.start()

            try:

                # The same caller-side copy is traced
                # for every algorithm.
                values = original[:]

                result = func(values)

                if mutates:
                    result = values

                current_bytes, peak_bytes = (
                    tracemalloc.get_traced_memory()
                )

            finally:

                tracemalloc.stop()

            if result != expected:

                raise RuntimeError(
                    f"{name} failed during memory measurement"
                )

            rows.append({

                "memory_repetition": repetition,

                "algorithm": name,

                "current_bytes": current_bytes,

                "peak_bytes": peak_bytes,

                "peak_MB": peak_bytes / (1024 * 1024)

            })

    return rows


# ============================================================
# 13. SUMMARY CALCULATIONS
# ============================================================

def medians_by_algorithm(rows, value_key):

    grouped = {}

    for row in rows:

        grouped.setdefault(
            row["algorithm"], []
        ).append(row[value_key])

    return {

        name: statistics.median(values)

        for name, values in grouped.items()

    }


def write_csv(path, rows):

    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()

        writer.writerows(rows)


def summarize(
    config,
    timing_rows,
    memory_rows
):

    times = medians_by_algorithm(
        timing_rows,
        "time_ms"
    )

    memory = medians_by_algorithm(
        memory_rows,
        "peak_MB"
    )

    row = {

        "size": config["size"],

        "low": config["low"],

        "high": config["high"],

        "pattern": config["pattern"]

    }

    # Add timing and memory results.
    for name, _, _ in ALGORITHMS:

        row[f"{name} median_ms"] = times[name]

        row[f"{name} peak_MB"] = memory[name]

    # Equal-digit-width speedup comparisons.
    # Speedup > 1 means Compartment Sort is faster.

    for digit in (1, 2, 3):

        r_name = f"Radix {digit}-bit"

        c_name = f"Compartment {digit}-bit"

        row[f"{digit}-bit speedup R/C"] = (
            times[r_name] / times[c_name]
        )

    return row


# ============================================================
# 14. MAIN EXPERIMENT
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Radix-sort benchmark suite"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a small pipeline test"
    )

    parser.add_argument(
        "--mode",
        choices=["sorting_only", "end_to_end"],
        default="sorting_only"
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=None
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS
    )

    parser.add_argument(
        "--memory-repeats",
        type=int,
        default=DEFAULT_MEMORY_REPEATS
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results")
    )

    args = parser.parse_args()

    if args.repeats < 1:

        parser.error(
            "--repeats must be >= 1"
        )

    if args.memory_repeats < 1:

        parser.error(
            "--memory-repeats must be >= 1"
        )

    if (
        args.sizes is not None
        and any(size < 1 for size in args.sizes)
    ):

        parser.error(
            "All --sizes values must be >= 1"
        )

    # --------------------------------------------------------
    # Run correctness tests first.
    # --------------------------------------------------------

    verify_algorithms()

    verify_edge_cases()

    # --------------------------------------------------------
    # Experiment configuration.
    # --------------------------------------------------------

    if args.quick:

        sizes = [
            1_000,
            5_000
        ]

        ranges = [
            (0, 255),
            (0, 1_000_000)
        ]

        patterns = [
            "random",
            "duplicates",
            "skewed"
        ]

        repeats = 3

        memory_repeats = 1

        output_dir = (
            args.output_dir / "quick_test"
        )

    else:

        sizes = (
            args.sizes
            if args.sizes
            else ARRAY_SIZES
        )

        ranges = VALUE_RANGES

        patterns = PATTERNS

        repeats = args.repeats

        memory_repeats = args.memory_repeats

        output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Reproducible random generators.
    # --------------------------------------------------------

    data_rng = random.Random(
        RANDOM_SEED
    )

    order_rng = random.Random(
        ORDER_SEED
    )

    timing_all = []

    memory_all = []

    summary_all = []

    total = (
        len(sizes)
        * len(ranges)
        * len(patterns)
    )

    index = 0

    # --------------------------------------------------------
    # Experiment information.
    # --------------------------------------------------------

    print("\n====================================")

    print("RADIX SORT BENCHMARK")

    print("====================================")

    print(f"\nMode: {args.mode}")

    print(
        f"Sizes: {sizes}; "
        f"timing repeats: {repeats}; "
        f"memory repeats: {memory_repeats}"
    )

    print(
        "Timing runs have tracemalloc disabled."
    )

    print(
        "Memory runs are measured separately."
    )

    print(
        "Algorithm order is shuffled "
        "on every timing repetition."
    )

    print(
        "Input-copy policy: one caller-side copy "
        "per algorithm."
    )

    if args.mode == "sorting_only":

        print(
            "Caller-side input copying is excluded "
            "from timing."
        )

    else:

        print(
            "Caller-side input copying is included "
            "in timing."
        )

    # --------------------------------------------------------
    # Run all experiment configurations.
    # --------------------------------------------------------

    for size in sizes:

        for low, high in ranges:

            for pattern in patterns:

                index += 1

                print(
                    f"\n[{index}/{total}] "
                    f"n={size}, "
                    f"range=({low},{high}), "
                    f"pattern={pattern}"
                )

                # Generate one dataset per configuration.
                original = make_input(
                    size,
                    low,
                    high,
                    pattern,
                    data_rng
                )

                expected = sorted(original)

                # Timing experiment.
                timing_rows = benchmark_timings(
                    original,
                    expected,
                    repeats,
                    args.mode,
                    order_rng
                )

                # Separate memory experiment.
                memory_rows = benchmark_memory(
                    original,
                    expected,
                    memory_repeats
                )

                config = {

                    "size": size,

                    "low": low,

                    "high": high,

                    "pattern": pattern

                }

                # Attach configuration information.
                for row in timing_rows:

                    row.update(config)

                for row in memory_rows:

                    row.update(config)

                # Generate summary.
                summary = summarize(
                    config,
                    timing_rows,
                    memory_rows
                )

                timing_all.extend(
                    timing_rows
                )

                memory_all.extend(
                    memory_rows
                )

                summary_all.append(
                    summary
                )

                # ------------------------------------------------
                # Print speedup results.
                # ------------------------------------------------

                print(
                    "\n  Speedup R/C: "
                    f"1-bit "
                    f"{summary['1-bit speedup R/C']:.3f}x | "
                    f"2-bit "
                    f"{summary['2-bit speedup R/C']:.3f}x | "
                    f"3-bit "
                    f"{summary['3-bit speedup R/C']:.3f}x"
                )

                # ------------------------------------------------
                # Print median timing results.
                # ------------------------------------------------

                print(
                    "\n  Median time (ms): "
                    f"R1 "
                    f"{summary['Radix 1-bit median_ms']:.3f}, "
                    f"C1 "
                    f"{summary['Compartment 1-bit median_ms']:.3f}, "
                    f"R2 "
                    f"{summary['Radix 2-bit median_ms']:.3f}, "
                    f"C2 "
                    f"{summary['Compartment 2-bit median_ms']:.3f}, "
                    f"R3 "
                    f"{summary['Radix 3-bit median_ms']:.3f}, "
                    f"C3 "
                    f"{summary['Compartment 3-bit median_ms']:.3f}"
                )

                # ------------------------------------------------
                # Print memory results.
                # ------------------------------------------------

                print(
                    "\n  Peak memory (MB): "
                    f"R3 "
                    f"{summary['Radix 3-bit peak_MB']:.3f}, "
                    f"C3 "
                    f"{summary['Compartment 3-bit peak_MB']:.3f}"
                )

                # ------------------------------------------------
                # Save after every configuration.
                # This protects results if a long run is interrupted.
                # ------------------------------------------------

                write_csv(
                    output_dir / "raw_timings.csv",
                    timing_all
                )

                write_csv(
                    output_dir / "raw_memory.csv",
                    memory_all
                )

                write_csv(
                    output_dir / "summary.csv",
                    summary_all
                )

    # --------------------------------------------------------
    # Experiment completed.
    # --------------------------------------------------------

    print("\n====================================")

    print("EXPERIMENT COMPLETED")

    print("====================================")

    print(
        f"\nResults directory: "
        f"{output_dir.resolve()}"
    )

    print("\nGenerated files:")

    print("1. raw_timings.csv")

    print("2. raw_memory.csv")

    print("3. summary.csv")


# ============================================================
# 15. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
