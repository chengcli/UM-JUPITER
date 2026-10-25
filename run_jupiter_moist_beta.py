"""Moist Jupiter beta-plane with bottom heating and top cooling."""

import argparse
import math
import os
from pathlib import Path

import torch
import yaml
from kintera import Kinetics, KineticsOptions, ThermoX
from snapy import Mesh, MeshOptions, kICY, kIV1
from paddle import setup_profile, evolve_kinetics
from paddle.forcings import BetaPlaneCoriolis

PERTURBATION_SEED = 0
VERTICAL_VELOCITY_PERTURBATION = 1.0e-2

def call_user_output(bvars: dict[str, torch.Tensor]):
    hydro_w = bvars["hydro_w"]
    return {"qtol": hydro_w[kICY:].sum(dim=0)}

@torch.no_grad()
def run_with(args):
    # load yaml configure
    infile = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    with infile.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    # create output directory
    options = MeshOptions.from_yaml(str(infile))
    device = torch.device(options.device_str())
    options.block().output_dir(str(output_dir))

    # construct mesh
    mesh = Mesh(options)
    mesh.to(device=device, dtype=torch.float64)
    blocks = list(mesh.blocks)

    # create forcing artifact folder for each rank separately
    rank = int(os.environ.get("RANK", "0"))
    artifact_dir = output_dir / "forcings" / f"rank{rank}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # create beta-plane forcing and register it at each rk stage
    problem = config["problem"]
    beta = 2.0 * float(problem["rotation_rate"]) / float(problem["radius"])
    coriolis_path = artifact_dir / "beta_plane.pt"
    torch.jit.script(BetaPlaneCoriolis(f0=0.0, beta=beta)).save(str(coriolis_path))
    mesh.set_user_stage_forcings([str(coriolis_path)])

    # create thermodynamics modules and user output function for each block
    thermodynamics = []
    for block in blocks:
        eos = block.module("hydro.eos")
        thermo_y = block.module("hydro.eos.thermo")
        thermo_x = ThermoX(thermo_y.options)
        thermo_x.to(device=device, dtype=torch.float64)
        thermodynamics.append((eos, thermo_x, thermo_y))
        block.set_user_output_func(call_user_output)

    # create kinetics modules
    kinet = Kinetics(KineticsOptions.from_yaml(str(infile)))
    kinet.to(device=device, dtype=torch.float64)

    if args.restart:
        block_vars, current_time = mesh.initialize_from_restart(str(Path(args.restart).resolve()))
    else:
        # 0 -> first block, 2 -> thermo_y
        species = thermodynamics[0][2].options.species()
        param = {"Ts": float(problem["Ts"]), "Ps": float(problem["Ps"]),
                 "Tmin": float(problem["Tmin"]),
                 "grav": -float(config["forcing"]["const-gravity"]["grav1"])}
        for name in species:
            param[f"x{name}"] = float(problem.get(f"x{name}", 0.0))

        # set initial profile with perturbation
        block_vars = []
        for block in blocks:
            hydro_w = setup_profile(block, param, method="pseudo-adiabat")
            layout = block.get_layout()
            rank = layout.options.rank()
            _, _, face_id = layout.loc_of(rank)
            generator = torch.Generator(device=hydro_w.device).manual_seed(
                PERTURBATION_SEED + face_id + rank * 6
            )
            hydro_w[kIV1].uniform_(
                -VERTICAL_VELOCITY_PERTURBATION,
                VERTICAL_VELOCITY_PERTURBATION,
                generator=generator,
            )
            block_vars.append({"hydro_w": hydro_w})
        block_vars, current_time = mesh.initialize(block_vars)

    # initialize cycle counts
    intg = mesh.module("block0.intg")
    cycle = blocks[0].cycle()
    if not args.restart:
        mesh.make_outputs(block_vars, current_time)
    mesh.set_cycle(cycle)

    while not intg.stop(cycle, current_time):
        dt = mesh.max_time_step(block_vars)
        mesh.print_cycle_info(block_vars, current_time, dt)

        # run rk-stages
        for stage in range(len(intg.stages)):
            mesh.forward(block_vars, dt, stage)

        # compute post-hydro primitives before applying moist chemistry.
        for bvars, (eos, thermo_x, thermo_y) in zip(block_vars, thermodynamics):
            bvars["hydro_w"] = eos.compute("U->W", (bvars["hydro_u"],))
            del_rho = evolve_kinetics(bvars["hydro_w"], eos,
                                      thermo_x, thermo_y, kinet, dt)
            bvars["hydro_u"][kICY:] += del_rho

        # check_redo will do a final hydro_u -> hydro_w
        err = mesh.check_redo(block_vars)
        if err > 0: # redo current step without advancing cycle
            continue
        if err < 0: # terminate
            break

        # all pass, advance a cycle
        cycle += 1
        mesh.set_cycle(cycle)
        current_time += dt

        mesh.make_outputs(block_vars, current_time)

    mesh.finalize(block_vars, current_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default=str(Path(__file__).with_name("jup_crm3d_H2O-NH3_beta.yaml")))
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("-r", "--restart", default="")
    run_with(parser.parse_args())
