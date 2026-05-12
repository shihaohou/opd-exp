"""
E1 on-policy v1 launch entrypoint.

Imports Delta-OPD wiring (agent loop subclass, worker patch, 4 losses) then
delegates to ``verl.trainer.main_ppo.main``. Use this in place of running
``python -m verl.trainer.main_ppo`` directly — verl's main_ppo would not
know to apply our patches.

Hydra picks up ``--config-path`` / ``--config-name`` from ``sys.argv``; the
launcher script (``scripts/run_e1_recipe.sh``) sets these to point at
``experiments/E1_filtered_delta_opd/configs``.
"""

from __future__ import annotations


def main() -> None:
    from experiments.E1_filtered_delta_opd.src.on_policy import enable

    enable()  # patches + registrations must happen before Hydra's main

    from verl.trainer.main_ppo import main as verl_main

    verl_main()


if __name__ == "__main__":
    main()
