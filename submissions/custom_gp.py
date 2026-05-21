"""
Custom differentiable global placer.

Minimizes the exact proxy cost: 1.0*WL + 0.5*Density + 0.5*Congestion
using PyTorch autograd and Adam optimizer.

Three differentiable loss terms:
  - HPWL:    log-sum-exp smooth approximation of half-perimeter wirelength
  - Density:  macro overlap penalty on a grid, blurred by Gaussian conv2d
  - RUDY:    differentiable RUDY congestion via log-sum-exp bounding boxes
"""

from __future__ import annotations

import time
import math
from typing import List, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from macro_place.benchmark import Benchmark


class CustomGP:
    """
    Differentiable global placer optimizing the full proxy cost.

    Usage:
        gp = CustomGP(benchmark)
        result = gp.optimize(init_positions, n_iters=2000)
    """

    def __init__(self, benchmark: "Benchmark"):
        N = benchmark.num_macros
        H = benchmark.num_hard_macros
        num_nets = benchmark.num_nets

        # --- Build padded pin data structures ---
        # net_pin_nodes[k] shape: [num_pins_in_net_k, 2]
        #   col0 = owner index: [0,H) hard, [H,N) soft, [N,N+P) ports
        #   col1 = pin slot: index into macro_pin_offsets[owner] for hard macros
        if len(benchmark.net_pin_nodes) == 0:
            # Fallback: derive from net_nodes (no pin-level offsets)
            pin_nodes_src = [
                torch.stack([nodes, torch.zeros_like(nodes)], dim=1)
                for nodes in benchmark.net_nodes
            ]
        else:
            pin_nodes_src = benchmark.net_pin_nodes

        max_pins = max((len(p) for p in pin_nodes_src), default=1)
        max_pins = max(max_pins, 1)

        # Padded owner indices: [num_nets, max_pins], fill=-1 for padding
        pin_owner = torch.full((num_nets, max_pins), -1, dtype=torch.long)
        # Padded pin offsets in microns: [num_nets, max_pins, 2]
        pin_offsets = torch.zeros(num_nets, max_pins, 2, dtype=torch.float32)
        # Valid-pin mask: [num_nets, max_pins]
        pin_mask = torch.zeros(num_nets, max_pins, dtype=torch.bool)

        has_offsets = len(benchmark.macro_pin_offsets) == H

        for k, pins in enumerate(pin_nodes_src):
            P = len(pins)
            if P == 0:
                continue
            owners = pins[:, 0].long()
            slots = pins[:, 1].long()
            pin_owner[k, :P] = owners
            pin_mask[k, :P] = True
            if has_offsets:
                for p in range(P):
                    o, s = int(owners[p]), int(slots[p])
                    if 0 <= o < H:
                        offsets_o = benchmark.macro_pin_offsets[o]
                        if s < len(offsets_o):
                            pin_offsets[k, p] = offsets_o[s]

        self.pin_owner = pin_owner      # [num_nets, max_pins]
        self.pin_offsets = pin_offsets  # [num_nets, max_pins, 2]
        self.pin_mask = pin_mask        # [num_nets, max_pins]
        self.net_weights = benchmark.net_weights.float()  # [num_nets]

        # Full position table: [num_macros + num_ports, 2]
        # Ports have owner indices [num_macros, num_macros + num_ports)
        self._port_positions = benchmark.port_positions.float()  # [P, 2], fixed
        self._num_macros = N

        # Canvas & grid
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.R = benchmark.grid_rows
        self.C = benchmark.grid_cols
        self.cell_w = self.cw / self.C
        self.cell_h = self.ch / self.R
        # Routing capacity per gcell (tracks in that gcell width/height)
        self.h_cap = float(benchmark.hroutes_per_micron * self.cell_h)
        self.v_cap = float(benchmark.vroutes_per_micron * self.cell_w)

        self.num_macros = N
        self.num_hard_macros = H
        self.macro_sizes = benchmark.macro_sizes.float()  # [N, 2]
        self.macro_fixed = benchmark.macro_fixed.bool()   # [N]

    # ------------------------------------------------------------------
    # Internal: build full position table (macros + ports on device)
    # ------------------------------------------------------------------

    def _full_pos(self, positions: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Concatenate macro positions with fixed port positions: [N+P, 2]."""
        ports = self._port_positions.to(device)
        if len(ports) == 0:
            return positions
        return torch.cat([positions, ports], dim=0)

    # ------------------------------------------------------------------
    # Phase 2A': Weighted-Average Wirelength (WA-WL) — smoother gradient
    # ------------------------------------------------------------------

    def wa_wl_loss(self, positions: torch.Tensor, gamma: float,
                   device: torch.device) -> torch.Tensor:
        """Differentiable weighted-average wirelength (Hsu et al., ePlace).

        For each net:
          x_max_smooth = Σ(x_i * exp(x_i/γ)) / Σ(exp(x_i/γ))
          x_min_smooth = Σ(x_i * exp(-x_i/γ)) / Σ(exp(-x_i/γ))
          WA-HPWL = (x_max - x_min) + (y_max - y_min)
        γ controls smoothing (small γ → sharper, large γ → smoother).
        Every pin contributes to the gradient (unlike argmax/argmin HPWL).
        """
        full_pos = self._full_pos(positions, device)
        owners = self.pin_owner.to(device)
        offsets = self.pin_offsets.to(device)
        mask = self.pin_mask.to(device)
        weights = self.net_weights.to(device)

        safe_owners = owners.clamp(min=0)
        pin_xy = full_pos[safe_owners] + offsets  # [N, P, 2]

        # Mask padding pins with -inf for stability in exp.
        NEG = -1e8
        x = pin_xy[..., 0]
        y = pin_xy[..., 1]
        x_masked = torch.where(mask, x, torch.full_like(x, NEG))
        y_masked = torch.where(mask, y, torch.full_like(y, NEG))

        # Stable softmax weights: subtract the max per net before exp.
        def _wa_max(v_masked):
            v_max = v_masked.detach().max(dim=1, keepdim=True).values
            w = torch.where(mask, torch.exp((v_masked - v_max) / gamma),
                            torch.zeros_like(v_masked))
            denom = w.sum(dim=1).clamp(min=1e-12)
            num = (w * v_masked).sum(dim=1)
            return num / denom  # [N]

        def _wa_min(v_masked):
            # Replace masked positions with +inf for min, via negation.
            neg = torch.where(mask, -v_masked, torch.full_like(v_masked, NEG))
            v_max = neg.detach().max(dim=1, keepdim=True).values
            w = torch.where(mask, torch.exp((neg - v_max) / gamma),
                            torch.zeros_like(neg))
            denom = w.sum(dim=1).clamp(min=1e-12)
            num = (w * neg).sum(dim=1)
            return -num / denom  # [N]

        x_max = _wa_max(x_masked); x_min = _wa_min(x_masked)
        y_max = _wa_max(y_masked); y_min = _wa_min(y_masked)
        wl = (x_max - x_min) + (y_max - y_min)
        valid = (mask.sum(dim=1) >= 2).float()
        return (wl * weights * valid).mean() / (self.cw + self.ch)

    # ------------------------------------------------------------------
    # Phase 2B': Hard-macro pairwise repulsion (keeps GP near-legal)
    # ------------------------------------------------------------------

    def hard_repulsion_loss(self, positions: torch.Tensor,
                             device: torch.device,
                             gap: float = 0.0) -> torch.Tensor:
        """Pairwise overlap penalty restricted to MOVABLE hard macros.

        Penalty per pair (i,j): max(0, sep_x - |dx|) * max(0, sep_y - |dy|)
        with sep_x = (w_i + w_j)/2 + gap. This is exactly the geometric
        overlap area — zero when separated, growing as macros interpenetrate.
        Quadratic in num_hard_macros, fine for IBM sizes (≤ ~2000).
        """
        H = self.num_hard_macros
        if H < 2:
            return torch.zeros((), device=device)
        pos_h = positions[:H]
        sizes_h = self.macro_sizes.to(device)[:H]
        half = sizes_h / 2

        dx = pos_h[:, 0].unsqueeze(1) - pos_h[:, 0].unsqueeze(0)
        dy = pos_h[:, 1].unsqueeze(1) - pos_h[:, 1].unsqueeze(0)
        sep_x = half[:, 0].unsqueeze(1) + half[:, 0].unsqueeze(0) + gap
        sep_y = half[:, 1].unsqueeze(1) + half[:, 1].unsqueeze(0) + gap

        ox = torch.clamp(sep_x - dx.abs(), min=0.0)
        oy = torch.clamp(sep_y - dy.abs(), min=0.0)
        overlap = ox * oy
        # zero out diagonal
        eye = torch.eye(H, device=device, dtype=overlap.dtype)
        overlap = overlap * (1.0 - eye)
        # mean over upper triangle to keep magnitude O(1)
        return overlap.sum() / max(1.0, H * (H - 1.0))

    # ------------------------------------------------------------------
    # Phase 2A: Differentiable HPWL via log-sum-exp
    # ------------------------------------------------------------------

    def hpwl_loss(self, positions: torch.Tensor, alpha: float,
                  device: torch.device) -> torch.Tensor:
        """
        Smooth HPWL via log-sum-exp bounding boxes over all nets.

        alpha controls smoothness: large alpha → smoother (biased high),
        small alpha → sharper approximation of true HPWL.
        """
        full_pos = self._full_pos(positions, device)  # [N+P, 2]

        owners = self.pin_owner.to(device)    # [num_nets, max_pins]
        offsets = self.pin_offsets.to(device)  # [num_nets, max_pins, 2]
        mask = self.pin_mask.to(device)        # [num_nets, max_pins]

        # Clamp -1 padding owners to 0 for safe gather, then mask out
        safe_owners = owners.clamp(min=0)
        pin_xy = full_pos[safe_owners] + offsets  # [num_nets, max_pins, 2]

        INF = 1e8
        pin_x = pin_xy[..., 0].masked_fill(~mask, -INF)
        neg_x = (-pin_xy[..., 0]).masked_fill(~mask, -INF)
        pin_y = pin_xy[..., 1].masked_fill(~mask, -INF)
        neg_y = (-pin_xy[..., 1]).masked_fill(~mask, -INF)

        # LSE bounding box per net: [num_nets]
        lse_xmax = alpha * torch.logsumexp(pin_x / alpha, dim=1)
        lse_xmin = -alpha * torch.logsumexp(neg_x / alpha, dim=1)
        lse_ymax = alpha * torch.logsumexp(pin_y / alpha, dim=1)
        lse_ymin = -alpha * torch.logsumexp(neg_y / alpha, dim=1)

        hpwl = (lse_xmax - lse_xmin) + (lse_ymax - lse_ymin)  # [num_nets]
        weights = self.net_weights.to(device)

        # Filter nets with at least 2 valid pins (single-pin nets have HPWL=0)
        valid_net = mask.sum(dim=1) >= 2
        # .mean() normalizes by net count so the loss is ~O(1) regardless of netlist size
        return (hpwl * weights * valid_net.float()).mean() / (self.cw + self.ch)

    # ------------------------------------------------------------------
    # Phase 2B'': ePlace-style electrostatic density (Poisson-solve density)
    # ------------------------------------------------------------------

    def eplace_density_loss(
        self,
        positions: torch.Tensor,
        sizes: torch.Tensor,
        device: torch.device,
        target_density: float = 0.7,
    ) -> torch.Tensor:
        """Electrostatic density loss à la ePlace (Cheng et al. 2018).

        Treats macro density as a 2D charge distribution ρ(x,y) and computes
        the electrostatic potential φ by solving the Poisson equation
        ∇²φ = -(ρ - bg) via 2D FFT. The loss is the half-energy
        ½ Σ ρ·φ. Gradients propagate through the FFT — the gradient on
        each macro position is the electric force E = -∇φ at that cell,
        giving smooth, global spreading forces (zero where density equals
        target; pushing outward where density exceeds target).

        bg = target_density makes the field vanish at the uniform
        ``target_density`` solution, so it's a valid equilibrium.
        """
        R, C = self.R, self.C
        cell_w, cell_h = self.cell_w, self.cell_h

        N = self.num_macros
        cx = positions[:N, 0]
        cy = positions[:N, 1]
        w = sizes[:N, 0]
        h = sizes[:N, 1]

        x_lo = (cx - w / 2) / cell_w
        x_hi = (cx + w / 2) / cell_w
        y_lo = (cy - h / 2) / cell_h
        y_hi = (cy + h / 2) / cell_h

        cell_x = torch.arange(C, dtype=torch.float32, device=device)
        cell_y = torch.arange(R, dtype=torch.float32, device=device)

        ox = torch.clamp(
            torch.minimum(x_hi.unsqueeze(1), cell_x + 1)
            - torch.maximum(x_lo.unsqueeze(1), cell_x),
            min=0.0,
        )  # [N, C] in cell units
        oy = torch.clamp(
            torch.minimum(y_hi.unsqueeze(1), cell_y + 1)
            - torch.maximum(y_lo.unsqueeze(1), cell_y),
            min=0.0,
        )  # [N, R]

        # Density on grid: fraction of cell occupied (sum over macros).
        density = (oy.unsqueeze(2) * ox.unsqueeze(1)).sum(0)  # [R, C]

        # Centre the charge distribution at the target density.
        rho = density - float(target_density)

        # Poisson solve via 2D FFT (with mirror padding to mimic Neumann BC).
        # Mirror-pad doubles each dim so the field vanishes at the boundary.
        rho_pad = torch.zeros(
            2 * R, 2 * C, device=device, dtype=rho.dtype
        )
        rho_pad[:R, :C] = rho
        rho_pad[:R, C:] = rho.flip(dims=(1,))
        rho_pad[R:, :C] = rho.flip(dims=(0,))
        rho_pad[R:, C:] = rho.flip(dims=(0, 1))

        rho_hat = torch.fft.fft2(rho_pad)

        # Build inverse Laplacian denominator in frequency space.
        # kx[i] = 2π * i / (2C * cell_w); but cell_w cancels out in the
        # adimensional formulation (positions are in cells via cx/cell_w),
        # so we use unit cell spacing for the inverse-Laplacian operator.
        kx = torch.fft.fftfreq(2 * C, d=1.0, device=device) * 2.0 * math.pi
        ky = torch.fft.fftfreq(2 * R, d=1.0, device=device) * 2.0 * math.pi
        kxx = kx.unsqueeze(0).pow(2)
        kyy = ky.unsqueeze(1).pow(2)
        denom = (kxx + kyy)
        denom[0, 0] = 1.0  # avoid div-by-zero at DC; we'll zero out below

        phi_hat = rho_hat / denom
        phi_hat[0, 0] = 0.0  # remove DC (constant offset)

        phi_pad = torch.fft.ifft2(phi_hat).real
        phi = phi_pad[:R, :C]  # un-pad

        # Electrostatic energy = ½ Σ ρ·φ; positive when charges concentrate.
        energy = 0.5 * (density * phi).sum()
        # Normalize by grid area so the value is O(1) regardless of size.
        return energy / (R * C)

    # ------------------------------------------------------------------
    # Phase 2B: Density loss with Gaussian blur for global spreading
    # ------------------------------------------------------------------

    def density_loss(self, positions: torch.Tensor, sizes: torch.Tensor,
                     sigma: float, device: torch.device,
                     target_density: float = 0.7) -> torch.Tensor:
        """
        Density overflow penalty on a grid, blurred with Gaussian conv2d.

        Uses ALL macros (hard + soft) so hard macros can't move into soft-macro
        clusters. Soft macros are frozen during optimization, so their density
        contribution is a constant that still produces the correct hard-macro gradient.
        """
        R, C = self.R, self.C
        cell_w, cell_h = self.cell_w, self.cell_h

        N = self.num_macros  # include all macros in density
        cx = positions[:N, 0]  # [N]
        cy = positions[:N, 1]
        w = sizes[:N, 0]
        h = sizes[:N, 1]

        # Macro extents in cell-grid coordinates (0-based fractional)
        x_lo = (cx - w / 2) / cell_w  # [H]
        x_hi = (cx + w / 2) / cell_w
        y_lo = (cy - h / 2) / cell_h
        y_hi = (cy + h / 2) / cell_h

        # Cell index arrays (left edges in grid coordinates)
        cell_x = torch.arange(C, dtype=torch.float32, device=device)  # [C]
        cell_y = torch.arange(R, dtype=torch.float32, device=device)  # [R]

        # Fractional overlap with each column: [N, C]
        ox = torch.clamp(
            torch.minimum(x_hi.unsqueeze(1), cell_x + 1)
            - torch.maximum(x_lo.unsqueeze(1), cell_x),
            min=0.0,
        ) * cell_w  # convert to μm overlap

        # Fractional overlap with each row: [N, R]
        oy = torch.clamp(
            torch.minimum(y_hi.unsqueeze(1), cell_y + 1)
            - torch.maximum(y_lo.unsqueeze(1), cell_y),
            min=0.0,
        ) * cell_h

        # Density grid [R, C]: Σ_macros (overlap_area / cell_area)
        cell_area = cell_w * cell_h
        density = (oy.unsqueeze(2) * ox.unsqueeze(1)).sum(0) / cell_area  # [R, C]

        # Gaussian blur for global spreading forces
        if sigma > 0.3:
            density = self._gaussian_blur_2d(density, sigma, device)

        overflow = torch.clamp(density - target_density, min=0.0)
        return overflow.pow(2).sum()

    def _gaussian_blur_2d(self, grid: torch.Tensor, sigma: float,
                           device: torch.device) -> torch.Tensor:
        k = max(3, (int(4 * sigma) | 1))  # odd, at least 3
        xs = torch.arange(k, dtype=torch.float32, device=device) - k // 2
        kernel_1d = torch.exp(-xs ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = (kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)).view(1, 1, k, k)
        return F.conv2d(grid.unsqueeze(0).unsqueeze(0), kernel_2d,
                        padding=k // 2)[0, 0]

    # ------------------------------------------------------------------
    # Phase 3: Differentiable RUDY congestion via LSE bounding boxes
    # ------------------------------------------------------------------

    def rudy_loss(self, positions: torch.Tensor, alpha: float,
                  device: torch.device,
                  chunk_size: int = 10_000) -> torch.Tensor:
        """
        RUDY routing demand with differentiable (LSE) bounding boxes.

        All pins contribute gradients via log-sum-exp.
        Large netlists are chunked to avoid OOM.
        """
        R, C = self.R, self.C
        cell_w, cell_h = self.cell_w, self.cell_h

        full_pos = self._full_pos(positions, device)
        owners = self.pin_owner.to(device)
        offsets = self.pin_offsets.to(device)
        mask = self.pin_mask.to(device)
        weights = self.net_weights.to(device)

        safe_owners = owners.clamp(min=0)
        pin_xy = full_pos[safe_owners] + offsets  # [num_nets, max_pins, 2]

        INF = 1e8
        pin_x = pin_xy[..., 0].masked_fill(~mask, -INF)
        neg_x = (-pin_xy[..., 0]).masked_fill(~mask, -INF)
        pin_y = pin_xy[..., 1].masked_fill(~mask, -INF)
        neg_y = (-pin_xy[..., 1]).masked_fill(~mask, -INF)

        # LSE bounding boxes: [num_nets]
        xmax = alpha * torch.logsumexp(pin_x / alpha, dim=1)
        xmin = -alpha * torch.logsumexp(neg_x / alpha, dim=1)
        ymax = alpha * torch.logsumexp(pin_y / alpha, dim=1)
        ymin = -alpha * torch.logsumexp(neg_y / alpha, dim=1)

        bbox_w = (xmax - xmin).clamp(min=1e-3)
        bbox_h = (ymax - ymin).clamp(min=1e-3)
        bbox_area = (bbox_w * bbox_h).clamp(min=1e-6)

        # Gcell grid edges in μm
        gcell_x_lo = torch.arange(C, dtype=torch.float32, device=device) * cell_w
        gcell_x_hi = gcell_x_lo + cell_w
        gcell_y_lo = torch.arange(R, dtype=torch.float32, device=device) * cell_h
        gcell_y_hi = gcell_y_lo + cell_h

        num_nets = pin_xy.shape[0]
        h_demand = torch.zeros(R, C, device=device, dtype=torch.float32)
        v_demand = torch.zeros(R, C, device=device, dtype=torch.float32)

        # Per-net H/V demand contribution (matches FastEval's RUDY formula):
        #   h_contrib[n] = ws[n] × cell_h / (bbox_h[n] × h_cap)  [util per gcell]
        #   v_contrib[n] = ws[n] × cell_w / (bbox_w[n] × v_cap)
        # These are ALREADY in utilization units — no further division by cap needed.
        # For a gcell fully inside net n's bbox: h_util += h_contrib[n].
        valid = (mask.sum(dim=1) >= 2).float()  # [num_nets]
        h_contrib = weights * valid * cell_h / (bbox_h * self.h_cap)  # [num_nets]
        v_contrib = weights * valid * cell_w / (bbox_w * self.v_cap)

        # Chunk to avoid [num_nets, R, C] OOM on large netlists
        for start in range(0, num_nets, chunk_size):
            end = min(start + chunk_size, num_nets)
            sl = slice(start, end)

            # Fractional gcell coverage in x/y: values in [0,1]
            # ox[n,c] / cell_w = fraction of column c covered by net n's bbox
            ox = torch.clamp(  # [chunk, C]
                torch.minimum(xmax[sl].unsqueeze(1), gcell_x_hi)
                - torch.maximum(xmin[sl].unsqueeze(1), gcell_x_lo),
                min=0.0,
            ) / cell_w  # normalise to [0,1]

            oy = torch.clamp(  # [chunk, R]
                torch.minimum(ymax[sl].unsqueeze(1), gcell_y_hi)
                - torch.maximum(ymin[sl].unsqueeze(1), gcell_y_lo),
                min=0.0,
            ) / cell_h

            # h_demand[r,c] += h_contrib[n] * x_cover[n,c] * y_cover[n,r]
            # v_demand[r,c] += v_contrib[n] * x_cover[n,c] * y_cover[n,r]
            h_c = h_contrib[sl].view(-1, 1, 1)  # [chunk,1,1]
            v_c = v_contrib[sl].view(-1, 1, 1)
            grid_cover = oy.unsqueeze(2) * ox.unsqueeze(1)  # [chunk,R,C]

            h_demand = h_demand + (h_c * grid_cover).sum(0)
            v_demand = v_demand + (v_c * grid_cover).sum(0)

        util = torch.maximum(h_demand, v_demand)  # already in utilization units

        # util^2 mean gives gradient to all cells; overflow mean focuses on congested cells.
        overflow = torch.clamp(util - 1.0, min=0.0)
        return util.pow(2).mean() + overflow.pow(2).mean() * 10.0

    # ------------------------------------------------------------------
    # Phase 1: Core optimization loop
    # ------------------------------------------------------------------

    def optimize(
        self,
        init_positions: torch.Tensor,
        n_iters: int = 2000,
        lr: float = 3e-3,
        wl_w: float = 1.0,
        density_w_start: float = 0.05,
        density_w_final: float = 0.5,
        rudy_w: float = 0.4,
        density_anneal_start: int = 50,
        density_ramp: float = 1.07,
        density_ramp_interval: int = 20,
        alpha_start: float = 0.5,
        sigma_start: float = 4.0,
        target_density: float = 0.7,
        device: Optional[torch.device] = None,
        log_every: int = 200,
    ) -> torch.Tensor:
        """
        Run Adam optimization on macro positions.

        Returns optimized positions (detached, on CPU).
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        positions = init_positions.clone().float().to(device)
        positions.requires_grad_(True)

        sizes = self.macro_sizes.to(device)
        # Freeze both explicitly fixed macros AND soft macros (SA re-places them later)
        soft_mask = torch.zeros(self.num_macros, dtype=torch.bool)
        soft_mask[self.num_hard_macros:] = True
        fixed = (self.macro_fixed | soft_mask).to(device)
        init_fixed = init_positions.float().to(device)

        opt = torch.optim.Adam([positions], lr=lr)

        density_w = density_w_start

        for it in range(n_iters):
            opt.zero_grad()

            # Ramp density weight every density_ramp_interval iters after warmup
            if it > density_anneal_start and it % density_ramp_interval == 0:
                density_w = min(density_w * density_ramp, density_w_final)

            progress = it / max(n_iters - 1, 1)
            sigma = sigma_start * (1.0 - progress) + 0.5 * progress
            alpha = alpha_start * (1.0 - 0.8 * progress)  # 0.5 → 0.1

            loss = (
                wl_w * self.hpwl_loss(positions, alpha, device)
                + density_w * self.density_loss(positions, sizes, sigma, device, target_density)
                + rudy_w * self.rudy_loss(positions, alpha, device)
            )

            loss.backward()

            # Zero out gradients for fixed macros before the optimizer step
            with torch.no_grad():
                positions.grad[fixed] = 0.0

            opt.step()

            # Clamp positions inside canvas bounds and restore fixed macros
            with torch.no_grad():
                half_w = sizes[:, 0] / 2
                half_h = sizes[:, 1] / 2
                positions[:, 0].clamp_(half_w, self.cw - half_w)
                positions[:, 1].clamp_(half_h, self.ch - half_h)
                positions[fixed] = init_fixed[fixed]

            if log_every > 0 and it % log_every == 0:
                print(
                    f"  [CustomGP] iter={it:4d}/{n_iters}"
                    f"  loss={loss.item():.4f}"
                    f"  density_w={density_w:.3f}"
                    f"  alpha={alpha:.3f}"
                    f"  sigma={sigma:.2f}"
                )

        return positions.detach().cpu()

    # ------------------------------------------------------------------
    # optimize_v2 — Nesterov + cosine lr + WA-WL + repulsion + proxy gate
    # ------------------------------------------------------------------

    def optimize_v2(
        self,
        init_positions: torch.Tensor,
        n_iters: int = 1500,
        lr_start: float = 4e-2,
        lr_end: float = 5e-4,
        momentum: float = 0.9,
        wl_w: float = 1.0,
        density_w_start: float = 0.03,
        density_w_final: float = 0.7,
        rudy_w: float = 0.6,
        repulsion_w: float = 1.5e-4,
        use_eplace_density: bool = True,
        eplace_w: float = 0.6,
        density_anneal_start: int = 40,
        density_ramp: float = 1.04,
        density_ramp_interval: int = 15,
        gamma_start: float = 8.0,
        gamma_end: float = 0.8,
        sigma_start: float = 4.0,
        sigma_end: float = 0.5,
        target_density: float = 0.7,
        proxy_gate_every: int = 0,
        fast_eval=None,
        device: Optional[torch.device] = None,
        log_every: int = 0,
    ) -> torch.Tensor:
        """Stronger differentiable GP loop.

        Differences from :meth:`optimize`:
          * Weighted-Average WL (WA-WL) instead of LSE HPWL — gradient on
            every pin proportional to its position; much smoother for
            late-stage refinement.
          * Hard-macro pairwise repulsion penalty pushes overlapping hard
            macros apart inside the GP, so legalization downstream becomes
            a cheap last-mile fix instead of a heavy basin-shift.
          * Nesterov-momentum SGD with cosine-annealed learning rate
            (lr_start → lr_end). This is the descent rule ePlace/RePlAce
            use; it provides much better convergence than Adam for the
            density+wirelength loss surface.
          * Optional proxy-monotone gate: every `proxy_gate_every` iters,
            if `fast_eval` is provided, evaluate the actual proxy; revert
            to the previous best if it regressed (cheap safety net).
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        positions = init_positions.clone().float().to(device)
        positions.requires_grad_(True)

        sizes = self.macro_sizes.to(device)
        # Move ALL macros that the benchmark marks movable (hard + soft).
        # Freezing soft macros artificially crippled the GP — soft macros
        # contribute most of the density grid, so spreading them out is
        # what produces DREAMPlace-quality global placement.
        fixed = self.macro_fixed.to(device)
        init_fixed = init_positions.float().to(device)

        opt = torch.optim.SGD([positions], lr=lr_start,
                              momentum=momentum, nesterov=True)

        density_w = density_w_start
        best_proxy = float("inf")
        best_snapshot = positions.detach().clone()

        for it in range(n_iters):
            opt.zero_grad()

            # cosine-annealed learning rate
            progress = it / max(n_iters - 1, 1)
            lr_now = lr_end + 0.5 * (lr_start - lr_end) * (
                1.0 + math.cos(math.pi * progress)
            )
            for pg in opt.param_groups:
                pg["lr"] = lr_now

            # ramp density weight after warmup
            if it > density_anneal_start and it % density_ramp_interval == 0:
                density_w = min(density_w * density_ramp, density_w_final)

            # anneal gamma (WA-WL smoothing) and sigma (density blur)
            gamma = gamma_end + (gamma_start - gamma_end) * (1.0 - progress)
            sigma = sigma_end + (sigma_start - sigma_end) * (1.0 - progress)

            loss = (
                wl_w * self.wa_wl_loss(positions, gamma, device)
                + rudy_w * self.rudy_loss(positions, gamma * 1.2, device)
                + repulsion_w * self.hard_repulsion_loss(positions, device)
            )
            if use_eplace_density:
                # ePlace electrostatic density: global, smooth spreading.
                loss = loss + eplace_w * self.eplace_density_loss(
                    positions, sizes, device, target_density
                )
            else:
                loss = loss + density_w * self.density_loss(
                    positions, sizes, sigma, device, target_density
                )
            loss.backward()

            with torch.no_grad():
                positions.grad[fixed] = 0.0
            opt.step()

            with torch.no_grad():
                half_w = sizes[:, 0] / 2
                half_h = sizes[:, 1] / 2
                positions[:, 0].clamp_(half_w, self.cw - half_w)
                positions[:, 1].clamp_(half_h, self.ch - half_h)
                positions[fixed] = init_fixed[fixed]

            if (
                proxy_gate_every > 0
                and fast_eval is not None
                and it > 0
                and it % proxy_gate_every == 0
            ):
                try:
                    cur = float(fast_eval.evaluate(positions.detach().cpu()))
                except Exception:
                    cur = float("inf")
                if cur < best_proxy:
                    best_proxy = cur
                    best_snapshot = positions.detach().clone()
                elif cur > best_proxy + 0.10:
                    # large regression — roll back
                    with torch.no_grad():
                        positions.copy_(best_snapshot)

            if log_every > 0 and it % log_every == 0:
                print(
                    f"  [CustomGP v2] iter={it:4d}/{n_iters} "
                    f"loss={loss.item():.4f} lr={lr_now:.4f} "
                    f"density_w={density_w:.3f} gamma={gamma:.2f} sigma={sigma:.2f}"
                )

        # Return the best snapshot if proxy gate populated it, else the
        # final positions.
        if best_proxy < float("inf"):
            return best_snapshot.detach().cpu()
        return positions.detach().cpu()

    # ------------------------------------------------------------------
    # Phase 4A: Coordinate Descent polish (gradient-free, fast_eval based)
    # ------------------------------------------------------------------

    def cd_polish(
        self,
        positions: torch.Tensor,
        benchmark: "Benchmark",
        fast_eval,
        n_rounds: int = 2,
    ) -> torch.Tensor:
        """
        Move each macro to the best of a 3×3 coarse grid around its current
        position, accepting moves that reduce fast_eval cost.
        """
        pos = positions.clone()
        H = benchmark.num_hard_macros
        movable = [i for i in range(H) if not bool(benchmark.macro_fixed[i])]
        cw, ch = self.cw, self.ch

        for _ in range(n_rounds):
            for i in movable:
                cx, cy = float(pos[i, 0]), float(pos[i, 1])
                w = float(benchmark.macro_sizes[i, 0])
                h = float(benchmark.macro_sizes[i, 1])
                step = max(w, h) * 0.5

                best_cost = fast_eval.evaluate(pos)
                best_xy = (cx, cy)

                for dx in (-step, 0.0, step):
                    for dy in (-step, 0.0, step):
                        if dx == 0.0 and dy == 0.0:
                            continue
                        nx = max(w / 2, min(cw - w / 2, cx + dx))
                        ny = max(h / 2, min(ch - h / 2, cy + dy))
                        pos[i, 0] = nx
                        pos[i, 1] = ny
                        cost = fast_eval.evaluate(pos)
                        if cost < best_cost:
                            best_cost = cost
                            best_xy = (nx, ny)

                pos[i, 0] = best_xy[0]
                pos[i, 1] = best_xy[1]

        return pos

    # ------------------------------------------------------------------
    # Phase 4B: Multi-seed runner — returns the best legalized placement
    # ------------------------------------------------------------------

    def run_seeds(
        self,
        seeds: List[torch.Tensor],
        benchmark: "Benchmark",
        plc,
        legalize_fn,
        oracle_fn,
        n_iters: int = 300,
        time_limit: float = 300.0,
        device: Optional[torch.device] = None,
        fast_eval=None,
        optimize_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Optimize each seed, legalize, score via oracle, return the best.

        Args:
            seeds:           list of [num_macros, 2] starting positions (must be legal)
            benchmark:       Benchmark object
            plc:             PlacementCost object (for oracle scoring)
            legalize_fn:     callable(positions, benchmark) → legal positions
            oracle_fn:       callable(positions, benchmark, plc) → dict with proxy_cost
            n_iters:         Adam iterations per seed
            time_limit:      wall-clock budget in seconds for all seeds combined
            device:          torch device (auto-detects CUDA if None)
            fast_eval:       optional FastEvaluator for CD polish step
            optimize_kwargs: extra kwargs passed to self.optimize()
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        best_cost = float("inf")
        best_pos = seeds[0]
        t0 = time.time()

        for idx, seed_pos in enumerate(seeds):
            elapsed = time.time() - t0
            if elapsed >= time_limit:
                print(f"  [CustomGP] time limit {time_limit:.0f}s reached after {idx} seeds")
                break

            remaining = time_limit - elapsed
            # Scale iters down if we're short on time
            iters = min(n_iters, max(50, int(n_iters * remaining / time_limit)))

            print(f"  [CustomGP] seed {idx + 1}/{len(seeds)}"
                  f"  iters={iters}  elapsed={elapsed:.1f}s")

            kw = dict(optimize_kwargs) if optimize_kwargs else {}
            result = self.optimize(seed_pos, n_iters=iters, device=device, **kw)

            # Restore soft macro positions to seed values — SA re-places them later.
            H = benchmark.num_hard_macros
            result[H:] = seed_pos[H:].float()

            if fast_eval is not None:
                result = self.cd_polish(result, benchmark, fast_eval, n_rounds=1)

            result = legalize_fn(result, benchmark)
            score = oracle_fn(result, benchmark, plc)["proxy_cost"]
            print(f"  [CustomGP] seed {idx + 1} proxy={score:.4f}")

            if score < best_cost:
                best_cost = score
                best_pos = result

        print(f"  [CustomGP] best proxy={best_cost:.4f}"
              f"  ({len(seeds)} seeds, {time.time() - t0:.1f}s)")
        return best_pos


# ------------------------------------------------------------------
# Convenience: build seed list from CT positions + optional Xplace result
# ------------------------------------------------------------------

def build_seeds(
    benchmark: "Benchmark",
    xplace_best: Optional[torch.Tensor] = None,
    noise_scales: tuple = (0.03, 0.08, 0.15),
    n_random: int = 2,
    rng: Optional[torch.Generator] = None,
) -> List[torch.Tensor]:
    """
    Build a diverse set of starting positions for CustomGP.

    Seeds (in priority order):
      1. Xplace best result (if provided)
      2. CT positions (from initial.plc)
      3. Noised CT positions (Gaussian noise at varying scales)
      4. Random positions uniformly within canvas
    """
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    ct = benchmark.macro_positions.clone().float()
    seeds: List[torch.Tensor] = []

    fixed = benchmark.macro_fixed

    if xplace_best is not None:
        s = xplace_best.clone().float()
        s[fixed] = ct[fixed]  # always restore fixed macros to their original positions
        seeds.append(s)

    seeds.append(ct.clone())

    for scale in noise_scales:
        noise = torch.zeros_like(ct)
        if rng is not None:
            noise[:, 0] = torch.normal(0, cw * scale, size=(len(ct),), generator=rng)
            noise[:, 1] = torch.normal(0, ch * scale, size=(len(ct),), generator=rng)
        else:
            noise[:, 0].normal_(0, cw * scale)
            noise[:, 1].normal_(0, ch * scale)
        noised = ct + noise
        # Clamp to canvas using macro half-sizes
        half_w = benchmark.macro_sizes[:, 0] / 2
        half_h = benchmark.macro_sizes[:, 1] / 2
        noised[:, 0].clamp_(half_w, cw - half_w)
        noised[:, 1].clamp_(half_h, ch - half_h)
        # Restore fixed macros
        fixed = benchmark.macro_fixed
        noised[fixed] = ct[fixed]
        seeds.append(noised)

    for _ in range(n_random):
        rand_pos = ct.clone()
        n = benchmark.num_macros
        half_w = benchmark.macro_sizes[:, 0] / 2
        half_h = benchmark.macro_sizes[:, 1] / 2
        if rng is not None:
            rx = torch.rand(n, generator=rng)
            ry = torch.rand(n, generator=rng)
        else:
            rx = torch.rand(n)
            ry = torch.rand(n)
        rand_pos[:, 0] = half_w + rx * (cw - 2 * half_w).clamp(min=0)
        rand_pos[:, 1] = half_h + ry * (ch - 2 * half_h).clamp(min=0)
        fixed = benchmark.macro_fixed
        rand_pos[fixed] = ct[fixed]
        seeds.append(rand_pos)

    return seeds
