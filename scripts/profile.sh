#!/bin/bash

# Profile fb_cpr_expl.py to find bottlenecks
# Usage: ./profile.sh

algo_seed=0
env_seed=0

echo "Profiling fb_cpr_final.py (first 3 episodes)..."

# Method 1: cProfile (detailed function-level profiling)
uv run python3 -m cProfile -o profile_output.prof crl/algos/fb_cpr_final.py \
    --env.domain_name dmc/walker \
    --env.task_list full \
    --env.seed ${env_seed} \
    --model.device cuda \
    --seed ${algo_seed} \
    --num_episodes 3 \
    --expl.epsilon 0.5 \
    --expl.num_z_samples 5 \
    --eval_freq 500 \
    --num_inference_samples 2000 \
    --no-do_eval \
    --no-use_wandb

echo "Generating human-readable profile report..."

# Generate sorted report by cumulative time
uv run python3 << 'EOF'
import pstats
from pstats import SortKey

# Load the profile data
p = pstats.Stats('profile_output.prof')

# Write full report to file
with open('profile_report.txt', 'w') as f:
    # Sort by cumulative time and print top 50 functions
    f.write("=" * 80 + "\n")
    f.write("TOP 50 FUNCTIONS BY CUMULATIVE TIME\n")
    f.write("=" * 80 + "\n")
    p.sort_stats(SortKey.CUMULATIVE)
    p.stream = f
    p.print_stats(50)

    # Sort by total time (time spent in function itself)
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("TOP 50 FUNCTIONS BY TOTAL TIME (excluding subcalls)\n")
    f.write("=" * 80 + "\n")
    p.sort_stats(SortKey.TIME)
    p.print_stats(50)

    # Show callers for expensive functions
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("CALLERS OF EXPENSIVE FUNCTIONS\n")
    f.write("=" * 80 + "\n")
    p.print_callers(20)

print("✓ Profile saved to profile_output.prof")
print("✓ Human-readable report saved to profile_report.txt")
print("\nTop 10 bottlenecks by cumulative time:")
p.sort_stats(SortKey.CUMULATIVE)
p.print_stats(10)
EOF

echo ""
echo "Profile complete! Check profile_report.txt for detailed analysis."
echo ""
echo "To view with visualization tools:"
echo "  - SnakeViz: uv run snakeviz profile_output.prof"
echo "  - gprof2dot: gprof2dot -f pstats profile_output.prof | dot -Tpng -o profile.png"

