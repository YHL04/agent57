

import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

import portpicker
import os
import time

from agent import Learner


def run_worker(rank,
               env_name,
               buffer_size,
               batch_size,
               burnin,
               rollout,
               n_step,
               n_cos,
               n_tau
               ):
    """
    Workers:
        - Learner.run()
        - Actor.run()

    Threads:
        - Learner.answer_requests(time.sleep(0.0001))
        - Learner.prepare_data(time.sleep(0.001))
        - ReplayBuffer.add_data(time.sleep(0.001))
        - ReplayBuffer.prepare_data(time.sleep(0.001))
        - ReplayBuffer.update_data(time.sleep(0.001))
        - ReplayBuffer.log_data(time.sleep(10))

    """

    if rank == 0:
        # create Learner in a remote location
        rpc.init_rpc("learner", rank=rank, world_size=2)

        learner_rref = rpc.remote(
            "learner",
            Learner,
            args=(env_name,
                  buffer_size,
                  batch_size,
                  n_cos,
                  n_tau,
                  n_step,
                  burnin,
                  rollout
                  ),
            timeout=0
        )
        learner_rref.remote().run()

        # Start training loop
        while True:
            time.sleep(1)

    else:
        # Create actor in a remote location
        rpc.init_rpc("actor", rank=rank, world_size=2)

    rpc.shutdown()


def main(env_name,
         buffer_size,
         batch_size,
         burnin,
         rollout,
         n_step,
         n_cos,
         n_tau
         ):
    # set localhost and port
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(portpicker.pick_unused_port())

    mp.spawn(
        run_worker,
        args=(env_name,
              buffer_size,
              batch_size,
              burnin,
              rollout,
              n_step,
              n_cos,
              n_tau
              ),
        nprocs=2,
        join=True
    )


if __name__ == "__main__":
    main(env_name="BreakoutDeterministic-v4",
         buffer_size=1_000_000,
         batch_size=64,
         burnin=0,
         rollout=10,
         n_step=1,
         n_cos=64,
         n_tau=64
         )
