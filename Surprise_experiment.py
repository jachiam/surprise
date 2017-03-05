# RL algorithm
from sandbox.surprisebased.algos.plus.trpo_plus import TRPO

# Exploration incentives
from exploration_bonuses.surprisal_bonus import SurprisalBonus
from exploration_bonuses.prediction_error_bonus import PredictionErrorBonus
from exploration_bonuses.approx_kl_div_n_step_bonus import ApproxKLNStepBonus


# Gym
from rllab.envs.gym_env import GymEnv
from sandbox.surprisebased.envs.normalized_atari_env import NormalizedAtariEnv

# Sparse reward tasks
# ---easier (classic control)
from sandbox.surprisebased.envs.mountain_car_env_x import MountainCarEnvX
from sandbox.surprisebased.envs.cartpole_swingup_env_x import CartpoleSwingupEnvX
# ---harder (locomotion)
from sandbox.surprisebased.envs.half_cheetah_env_x import HalfCheetahEnvX
from sandbox.surprisebased.envs.swimmer_env_x import SwimmerEnvX
from rllab.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv
from rllab.envs.mujoco.gather.ant_gather_env import AntGatherEnv
from rllab.envs.mujoco.maze.swimmer_maze_env import SwimmerMazeEnv
from rllab.envs.mujoco.maze.ant_maze_env import AntMazeEnv


# Baselines
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
import lasagne.nonlinearities as NL

# Policy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer

# Instrumentation
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
#from custom_plotter import *



stub(globals())


#===========================#
# SPARSE REWARD EXPERIMENTS #
#===========================#

mc_experiment = {
            'env_name': 'MountainCarEnvX',
            'task_type': 'classic', 
            'env_call': MountainCarEnvX, 
            'normalize_env': False
            }

cps_experiment = {
            'env_name': 'CartpoleSwingupEnvX',
            'task_type': 'classic', 
            'env_call': CartpoleSwingupEnvX, 
            'normalize_env': False
            }

hc_experiment = {
            'env_name': 'HalfCheetahEnvX',
            'task_type': 'locomotion', 
            'env_call': HalfCheetahEnvX, 
            'normalize_env': True
            }

swim_experiment = {
            'env_name': 'SwimmerEnvX',
            'task_type': 'locomotion', 
            'env_call': SwimmerEnvX, 
            'normalize_env': True
            }

sg_experiment = {
            'env_name': 'SwimmerGather',
            'task_type': 'heirarchical', 
            'env_call': SwimmerGatherEnv, 
            'normalize_env': True
            }


#===================#
# ATARI EXPERIMENTS #
#===================#

ven_experiment = {
            'env_name': 'Venture-ram-v0',
            'task_type': 'atari'
            }

bh_experiment = {
            'env_name': 'BankHeist-ram-v0',
            'task_type': 'atari'
            }

fw_experiment = {
            'env_name': 'Freeway-ram-v0',
            'task_type': 'atari'
            }

pong_experiment = {
            'env_name': 'Pong-ram-v0',
            'task_type': 'atari'
            }



experiment = cps_experiment
experiment_name = 'TRPO-surprisal-demo-' + experiment['env_name']

for j in range(5):

    task_type = experiment['task_type']
    if task_type=='atari':
        env = NormalizedAtariEnv(GymEnv(experiment['env_name'],record_video=False))
    else:
        env = experiment['env_call']()
        if experiment['normalize_env']:
            env = normalize(env)


    if task_type == 'classic':

        trpo_max_path_length = 500
        trpo_batch_size = 5000
        trpo_subsample_factor = 1
        trpo_step_size = 0.01
        expl_lambda = 0.001

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(32,),
        )

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args={
                    'hidden_sizes':(32,),
                    'hidden_nonlinearity': NL.tanh,
                    'learn_std':False,
                    'step_size':0.01,
                    'optimizer':ConjugateGradientOptimizer(subsample_factor=trpo_subsample_factor)
                    }
        )
            
    elif task_type == 'locomotion':

        trpo_max_path_length = 500
        trpo_batch_size = 5000
        trpo_subsample_factor = 1
        trpo_step_size = 0.05
        expl_lambda = 0.001

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64,32),
        )
        
        baseline = LinearFeatureBaseline(env_spec=env.spec)
            

    elif task_type == 'heirarchical':

        trpo_max_path_length = 500
        trpo_batch_size = 50000
        trpo_subsample_factor = 0.1
        trpo_step_size = 0.01
        expl_lambda = 0.0001

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(64,32),
        )
        
        baseline = LinearFeatureBaseline(env_spec=env.spec)
            

    elif task_type == 'atari':

        trpo_max_path_length = 7000
        trpo_batch_size = 50000
        trpo_subsample_factor = 0.2
        trpo_step_size = 0.01
        expl_lambda = 0.005

        policy = CategoricalMLPPolicy(
            env_spec = env.spec,
            hidden_sizes=(64,32),
        )
        
        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args={
                    'hidden_sizes':(64,32),
                    'hidden_nonlinearity': NL.tanh,
                    'learn_std':False,
                    'step_size':0.01,
                    'optimizer':ConjugateGradientOptimizer(subsample_factor=trpo_subsample_factor)
                    }
        )
        


    dynamics_batch_size = 5000 #10000
    dynamics_replay_size = 5000000 #1000000
    dynamics_hidden_sizes = (64,64)#(32,)
    dynamics_step_size = 0.001#0.01
    dynamics_subsample_factor = 1 #0.5
    dynamics_weight_decay = 1


    sur = SurprisalBonus(
            env.spec,
            use_grads=False,
            normalize_bonus=False,
            use_replay_pool=True,
            weight_decay=dynamics_weight_decay,
            batch_size=dynamics_batch_size,
            max_pool_size=dynamics_replay_size,
            hidden_sizes=dynamics_hidden_sizes,
            hidden_nonlinearity=NL.tanh,
            regressor_args={
                'use_trust_region':True,
                'step_size':dynamics_step_size,
                'optimizer':ConjugateGradientOptimizer(subsample_factor=dynamics_subsample_factor)
                }
    )

    pred = PredictionErrorBonus(
            env.spec,
            normalize_bonus=False,
            use_replay_pool=True,
            weight_decay=dynamics_weight_decay,
            batch_size=dynamics_batch_size,
            max_pool_size=dynamics_replay_size,
            hidden_sizes=dynamics_hidden_sizes,
            hidden_nonlinearity=NL.tanh,
            regressor_args={
                'use_trust_region':True,
                'step_size':dynamics_step_size,
                'optimizer':ConjugateGradientOptimizer(subsample_factor=dynamics_subsample_factor)
                },
            use_square_error=True,
    )


    akln = ApproxKLNStepBonus(
            env.spec,
            lag_steps=1,
            use_replay_pool=True,
            weight_decay=dynamics_weight_decay,
            batch_size=dynamics_batch_size,
            max_pool_size=dynamics_replay_size,
            hidden_sizes=dynamics_hidden_sizes,
            hidden_nonlinearity=NL.tanh,
            regressor_args={
                'use_trust_region':True,
                'step_size':dynamics_step_size,
                'optimizer':ConjugateGradientOptimizer(subsample_factor=dynamics_subsample_factor)
                }
    )


    algo = TRPO(
        env=env,
        exploration_bonus=sur,
        #exploration_bonus=pred,
        #exploration_bonus=akln,
        exploration_lambda=expl_lambda,
        normalize_bonus=True,
        nonnegative_bonus_mean=False, #True,
        all_paths=True,
        use_bonus_in_baseline=False,
        policy=policy,
        baseline=baseline,
        batch_size=trpo_batch_size,
        max_path_length=trpo_max_path_length,
        n_itr=500,
        discount=0.995,
        gae_lambda=0.95,
        step_size=trpo_step_size,
        min_num_paths=0,
        optimizer=ConjugateGradientOptimizer(subsample_factor=trpo_subsample_factor),
        #plot=True,
    )

    run_experiment_lite(
        algo.train(),
        n_parallel=4,
        snapshot_mode="last",
        seed=j,
        exp_prefix=experiment_name,
        #mode="ec2",
        #plot=True,
    )
