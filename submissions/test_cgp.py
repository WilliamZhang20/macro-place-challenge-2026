"""
Standalone test: Xplace best seed → CustomGP → cost breakdown comparison.

Run inside Docker:
  python3 submissions/koral/test_cgp.py -b ibm01
"""

import argparse, sys, os, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, '/challenge')

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--benchmark', default='ibm01')
parser.add_argument('--seed',  type=int,   default=58,   help='Xplace random seed (default: best ibm01 winner)')
parser.add_argument('--noise', type=float, default=0.15, help='Xplace noise ratio')
parser.add_argument('--cgp-iters', type=int, default=300)
args = parser.parse_args()

from macro_place.loader import load_benchmark
from macro_place.objective import compute_proxy_cost

BENCH_DIR = f'/challenge/external/MacroPlacement/Testcases/ICCAD04/{args.benchmark}'
print(f'\n=== CustomGP test: {args.benchmark} | Xplace seed={args.seed} noise={args.noise} ===\n')

benchmark, plc = load_benchmark(
    f'{BENCH_DIR}/netlist.pb.txt',
    f'{BENCH_DIR}/initial.plc',
    name=args.benchmark,
)
print(benchmark)

def cost_line(label, pos):
    r = compute_proxy_cost(pos, benchmark, plc)
    print(f'  {label:30s}  proxy={r["proxy_cost"]:.4f}  '
          f'wl={r["wirelength_cost"]:.4f}  '
          f'dens={r["density_cost"]:.4f}  '
          f'cong={r["congestion_cost"]:.4f}  '
          f'overlaps={r["overlap_count"]:.0f}')
    return r

# ── CT baseline ──────────────────────────────────────────────────────────────
cost_line('CT baseline', benchmark.macro_positions)

# ── Xplace (single seed) ─────────────────────────────────────────────────────
from placer import KoralPlacer
P = KoralPlacer(seed=args.seed, sa_time_budget=0)

xplace_home = os.environ.get('XPLACE_HOME', '/opt/xplace')
xplace_main = os.path.join(xplace_home, 'main.py')
if not os.path.isfile(xplace_main):
    print(f'Xplace not found at {xplace_main}')
    sys.exit(1)

from bookshelf import write_bookshelf
import subprocess, tempfile, shutil, numpy as np

tmpdir = tempfile.mkdtemp(prefix='cgp_test_')
try:
    write_bookshelf(benchmark, tmpdir)
    ct = benchmark.macro_positions.clone()
    n_hard = benchmark.num_hard_macros
    noised = ct.clone()
    rng = torch.Generator(); rng.manual_seed(args.seed)
    noise_x = torch.randn(n_hard, generator=rng) * args.noise * benchmark.canvas_width
    noise_y = torch.randn(n_hard, generator=rng) * args.noise * benchmark.canvas_height
    noised[:n_hard, 0] = (ct[:n_hard, 0] + noise_x).clamp(
        benchmark.macro_sizes[:n_hard,0]/2, benchmark.canvas_width  - benchmark.macro_sizes[:n_hard,0]/2)
    noised[:n_hard, 1] = (ct[:n_hard, 1] + noise_y).clamp(
        benchmark.macro_sizes[:n_hard,1]/2, benchmark.canvas_height - benchmark.macro_sizes[:n_hard,1]/2)

    # Write noised starting positions into bookshelf .pl
    from bookshelf import _write_pl
    _write_pl(noised, benchmark, tmpdir)

    cmd = [
        'python3', xplace_main,
        '--dataset_root', tmpdir,
        '--dataset', args.benchmark,
        '--stop_overflow', '0.05',
        '--inner_iter', '8000',
        '--mixed_size', 'True',
        '--noise_ratio', '0.0',   # noise already applied
        '--seed', str(args.seed),
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    xplace_time = time.time() - t0

    if result.returncode != 0:
        print(f'Xplace failed:\n{result.stderr[-500:]}')
        sys.exit(1)

    # Parse Xplace output positions
    from bookshelf import read_xplace_result
    xpl_pos = read_xplace_result(tmpdir, benchmark)
    xpl_legal = P._legalize_hard(xpl_pos, benchmark)
    print(f'\nXplace ({xplace_time:.0f}s):')
    cost_line(f'Xplace seed={args.seed} noise={args.noise}', xpl_pos)
    cost_line('Xplace legalized', xpl_legal)

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

# ── CustomGP ─────────────────────────────────────────────────────────────────
if len(benchmark.net_pin_nodes) == 0:
    print('\nnet_pin_nodes empty — CustomGP skipped')
    sys.exit(0)

from custom_gp import CustomGP
gp = CustomGP(benchmark)

kw = dict(
    wl_w=0.5, density_w_start=0.5, density_w_final=0.5,
    rudy_w=0.5,
    density_anneal_start=9999, density_ramp=1.0, density_ramp_interval=9999,
    sigma_start=2.0, alpha_start=0.3, lr=5e-4,
    log_every=100,
)

print(f'\nRunning CustomGP ({args.cgp_iters} iters) on Xplace legalized...')
t0 = time.time()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'  device: {device}')

cgp_raw = gp.optimize(xpl_legal.clone(), n_iters=args.cgp_iters, device=device, **kw)
cgp_raw[benchmark.num_hard_macros:] = xpl_legal[benchmark.num_hard_macros:].float()
cgp_legal = P._legalize_hard(cgp_raw, benchmark)
cgp_time = time.time() - t0

print(f'\nResults ({cgp_time:.0f}s):')
r_xpl  = cost_line('Xplace legalized (baseline)', xpl_legal)
r_cgp  = cost_line('CustomGP result', cgp_legal)

print(f'\n  Delta proxy:  {r_cgp["proxy_cost"]  - r_xpl["proxy_cost"]:+.4f}')
print(f'  Delta wl:     {r_cgp["wirelength_cost"]  - r_xpl["wirelength_cost"]:+.4f}')
print(f'  Delta dens:   {r_cgp["density_cost"]  - r_xpl["density_cost"]:+.4f}')
print(f'  Delta cong:   {r_cgp["congestion_cost"] - r_xpl["congestion_cost"]:+.4f}')
print()
if r_cgp["proxy_cost"] < r_xpl["proxy_cost"]:
    print('  ✓ CustomGP improved proxy cost')
else:
    print('  ✗ CustomGP did not improve proxy cost')
