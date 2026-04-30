import argparse
from dataclasses import dataclass

import torch
import yaml

from paddle import setup_profile
from snapy import Mesh, MeshOptions, kIDN, kIPR, kIV1
from snapy import EquationOfState

from jupiter_minichem import (
    EquilibriumTable,
    advance_minichem,
    build_pyminichem,
    initialize_scalar_mmr,
)


@dataclass(frozen=True)
class BlockContext:
    eos: EquationOfState


def call_user_output(bvars: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    if "scalar_r" in bvars:
        out["qtol"] = bvars["scalar_r"].sum(dim=0)
    return out


def make_problem_params(config: dict, species: list[str]) -> dict[str, float]:
    param = {
        "Ts": float(config["problem"]["Ts"]),
        "Ps": float(config["problem"]["Ps"]),
        "grav": -float(config["forcing"]["const-gravity"]["grav1"]),
        "Tmin": float(config["problem"]["Tmin"]),
    }
    for name in species:
        param[f"x{name}"] = float(config["problem"].get(f"x{name}", 0.0))
    return param


def ensure_scalar_state(block_vars: list[dict[str, torch.Tensor]]) -> None:
    for bvars in block_vars:
        if "scalar_r" not in bvars or "scalar_s" in bvars:
            continue
        if "hydro_u" in bvars:
            rho = bvars["hydro_u"][kIDN]
        else:
            rho = bvars["hydro_w"][kIDN]
        bvars["scalar_s"] = rho.unsqueeze(0) * bvars["scalar_r"]


def run_with(args: argparse.Namespace) -> None:
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    options = MeshOptions.from_yaml(args.config)
    options.block().output_dir(args.output_dir)
    mesh = Mesh(options)

    if torch.cuda.is_available() and options.block().layout().backend() == "nccl":
        device = torch.device(options.device_str())
    else:
        device = torch.device("cpu")

    mesh.to(device)

    block_contexts = [
        BlockContext(eos=block.module("hydro.eos"))
        for block in mesh.blocks
    ]

    if args.restart:
        block_vars, current_time = mesh.initialize_from_restart(args.restart)
    else:
        eq_table = EquilibriumTable.from_resource()
        dry_species = mesh.blocks[0].module("hydro.eos.thermo").options.species()
        param = make_problem_params(config, dry_species)
        block_vars = []

        for block, ctx in zip(mesh.blocks, block_contexts):
            hydro_w = setup_profile(block, param, method="pseudo-adiabat")
            hydro_w[kIV1] += 0.1 * torch.rand_like(hydro_w[kIV1])

            temp = ctx.eos.compute("W->T", (hydro_w,))
            pres = hydro_w[kIPR]
            scalar_r = initialize_scalar_mmr(eq_table, temp, pres)
            block_vars.append({"hydro_w": hydro_w, "scalar_r": scalar_r})

        block_vars, current_time = mesh.initialize(block_vars)

    ensure_scalar_state(block_vars)

    for block in mesh.blocks:
        block.set_user_output_func(call_user_output)

    mc = build_pyminichem()

    intg = mesh.module("block0.intg")
    cycle = 0
    if not args.restart:
        mesh.make_outputs(block_vars, current_time)

    while not intg.stop(cycle, current_time):
        cycle += 1
        mesh.set_cycle(cycle)

        dt = mesh.max_time_step(block_vars)
        mesh.print_cycle_info(block_vars, current_time, dt)
        scalar_s_prev = [bvars["scalar_s"].clone() for bvars in block_vars]

        for stage in range(len(intg.stages)):
            mesh.forward(block_vars, dt, stage)

        redo_vars = [
            {name: value for name, value in bvars.items() if name != "scalar_s"}
            for bvars in block_vars
        ]
        err = mesh.check_redo(redo_vars)
        if err > 0:
            for bvars, scalar_s0 in zip(block_vars, scalar_s_prev):
                bvars["scalar_s"].copy_(scalar_s0)
                bvars["scalar_r"] = scalar_s0 / bvars["hydro_u"][kIDN].unsqueeze(0)
            continue
        if err < 0:
            break

        for bvars, ctx in zip(block_vars, block_contexts):
            temp = ctx.eos.compute("W->T", (bvars["hydro_w"],))
            pres = bvars["hydro_w"][kIPR]
            rho = bvars["hydro_u"][kIDN]
            advance_minichem(mc, temp, pres, bvars["scalar_s"], rho, dt)
            bvars["scalar_r"] = bvars["scalar_s"] / rho.unsqueeze(0)

        current_time += dt
        mesh.make_outputs(block_vars, current_time)

    mesh.finalize(block_vars, current_time)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dry-Jupiter MiniChem transport.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        default="jupiter_crm_dry_minichem.yaml",
        help="Input YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
        default="output",
        help="Output directory",
    )
    parser.add_argument(
        "-r",
        "--restart",
        type=str,
        required=False,
        default="",
        help="Restart from restart dump.",
    )
    run_with(parser.parse_args())


if __name__ == "__main__":
    main()
