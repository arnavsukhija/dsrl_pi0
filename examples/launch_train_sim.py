import argparse
import sys
from examples.train_sim import main
from jaxrl2.utils.launch_util import parse_training_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, help='Random seed.', type=int)
    parser.add_argument('--launch_group_id', default='', help='group id used to group runs on wandb.')
    parser.add_argument('--eval_episodes', default=10,help='Number of episodes used for evaluation.', type=int)
    parser.add_argument('--env', default='libero', help='name of environment')
    parser.add_argument('--log_interval', default=1000, help='Logging interval.', type=int)
    parser.add_argument('--eval_interval', default=5000, help='Eval interval.', type=int)
    parser.add_argument('--checkpoint_interval', default=-1, help='checkpoint interval.', type=int)
    parser.add_argument('--batch_size', default=16, help='Mini batch size.', type=int)
    parser.add_argument('--max_steps', default=int(1e6), help='Number of training steps.', type=int)
    parser.add_argument('--add_states', default=1, help='whether to add low-dim states to the obervations', type=int)
    parser.add_argument('--wandb_project', default='cql_sim_online', help='wandb project')
    parser.add_argument('--start_online_updates', default=1000, help='number of steps to collect before starting online updates', type=int)
    parser.add_argument('--algorithm', default='pixel_sac', help='type of algorithm')
    parser.add_argument('--prefix', default='', help='prefix to use for wandb')
    parser.add_argument('--suffix', default='', help='suffix to use for wandb')
    parser.add_argument('--multi_grad_step', default=1, help='Number of graident steps to take per environment step, aka UTD', type=int)
    parser.add_argument('--resize_image', default=-1, help='the size of image if need resizing', type=int)
    parser.add_argument('--query_freq', default=-1, help='query frequency (only used when switch_cost_wrapper=0)', type=int)
    parser.add_argument('--libero_suite', default='libero_90', help='LIBERO benchmark suite to use when --env libero.')
    parser.add_argument('--libero_task_id', default=57, help='LIBERO task id within the selected benchmark suite.', type=int)

    # TARC-style adaptive replanning options
    parser.add_argument('--switch_cost_wrapper', default=0, type=int,
                        help='If 1, agent emits noise + pseudo_time per decision and pays a switch cost for each pi0 query.')
    parser.add_argument('--switch_cost', default=0.0, type=float,
                        help='Constant cost subtracted from the aggregated reward at every pi0 re-query.')
    parser.add_argument('--min_time_between_switches', default=1, type=int,
                        help='Minimum number of env steps to consume from a chunk before re-querying pi0.')
    parser.add_argument('--max_time_between_switches', default=20, type=int,
                        help='Maximum number of env steps to consume from a chunk before re-querying pi0 (must be <= 50 chunk length).')
    parser.add_argument('--time_as_part_of_state', default=1, type=int,
                        help='If 1, append normalized elapsed episode time to the SAC observation state.')
    parser.add_argument('--eval_mode', default='hybrid',
                        choices=['hybrid', 'fully_deterministic', 'fully_random'],
                        help='Eval-time action sampling: hybrid (noise=sample, K=mode), '
                             'fully_deterministic (both=mode), or fully_random (both=sample). '
                             'Only affects perform_control_eval_switchcost.')

    train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr= 3e-4,
        temp_lr=3e-4,
        dyn_ent_lr=3e-4,
        init_dyn_ent_temperature=1.0,
        model_lr=3e-4,
        model_wd=0.0,
        model_hidden_dims=(256, 256),
        num_model_heads=5,
        model_noise_var=1.0,
        predict_reward=True,
        predict_diff=True,
        backup_entropy=True,
        model_obs_key="state",
        obs_dim=64,
        ensemble_disagreement_modalities="",
        mask_expl_critic=False,
        tactile_hidden_dims=(256, 256),
        mask_touch=False,
        hidden_dims= (128, 128, 128),
        cnn_features= (32, 32, 32, 32),
        cnn_strides= (2, 1, 1, 1),
        cnn_padding= 'VALID',
        latent_dim= 50,
        discount= 0.999,
        tau= 0.005,
        critic_reduction = 'mean',
        dropout_rate=0.0,
        aug_next=1,
        use_bottleneck=True,
        encoder_type='small',
        encoder_norm='group',
        use_spatial_softmax=True,
        softmax_temperature=-1,
        target_entropy='auto',
        num_qs=10,
        action_magnitude=1.0,
        num_cameras=1,
        pseudo_time_init_bias=0.0,
        explore_until=300000,
        agent_update_period=1,
        expl_agent_update_period=1,
        ensemble_update_period=1,
        )

    variant, args = parse_training_args(train_args_dict, parser)
    print(variant)
    main(variant)
    sys.exit()
    
