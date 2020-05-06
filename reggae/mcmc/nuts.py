from tensorflow_probability import mcmc as tfpmcmc
import tensorflow as tf
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.math.generic import log_add_exp
from tensorflow_probability.python.mcmc.internal import leapfrog_integrator as leapfrog_impl
import numpy as np

# from tensorflow_probability.python.mcmc.nuts import *
MomentumStateSwap = tfpmcmc.nuts.MomentumStateSwap
NUTSKernelResults = tfpmcmc.nuts.NUTSKernelResults
compute_hamiltonian = tfpmcmc.nuts.compute_hamiltonian
has_not_u_turn = tfpmcmc.nuts.has_not_u_turn
TreeDoublingMetaState = tfpmcmc.nuts.TreeDoublingMetaState
TreeDoublingStateCandidate = tfpmcmc.nuts.TreeDoublingStateCandidate
TREE_COUNT_DTYPE = tfpmcmc.nuts.TREE_COUNT_DTYPE
MULTINOMIAL_SAMPLE = tfpmcmc.nuts.MULTINOMIAL_SAMPLE     
GENERALIZED_UTURN = tfpmcmc.nuts.MULTINOMIAL_SAMPLE 
_rightmost_expand_to_rank = tfpmcmc.nuts._rightmost_expand_to_rank


class NoUTurnSampler(tfpmcmc.NoUTurnSampler):
  def bootstrap_results(self, init_state):
    """Creates initial `previous_kernel_results` using a supplied `state`."""
    with tf.name_scope(self.name + '.bootstrap_results'):
      if not tf.nest.is_nested(init_state):
        init_state = [init_state]
      # Padding the step_size so it is compatable with the states
      step_size = self.step_size
      if len(step_size) == 1:
        step_size = step_size * len(init_state)
      if len(step_size) != len(init_state):
        raise ValueError('Expected either one step size or {} (size of '
                         '`init_state`), but found {}'.format(
                             len(init_state), len(step_size)))

      dummy_momentum = [tf.ones_like(state) for state in init_state]

      def _init(shape_and_dtype):
        """Allocate TensorArray for storing state and momentum."""
        return [  # pylint: disable=g-complex-comprehension
            prefer_static.zeros(
                prefer_static.concat([[max(self._write_instruction) + 1], s],
                                     axis=0),
                dtype=d) for (s, d) in shape_and_dtype
        ]

      get_shapes_and_dtypes = lambda x: [(prefer_static.shape(x_), x_.dtype)  # pylint: disable=g-long-lambda
                                         for x_ in x]
      momentum_state_memory = MomentumStateSwap(
          momentum_swap=_init(get_shapes_and_dtypes(dummy_momentum)),
          state_swap=_init(get_shapes_and_dtypes(init_state)))
      [
          _,
          _,
          current_target_log_prob,
          current_grads_log_prob,
      ] = leapfrog_impl.process_args(self.target_log_prob_fn(self.all_states_hack), dummy_momentum,
                                     init_state)

      return NUTSKernelResults(
          target_log_prob=current_target_log_prob,
          grads_target_log_prob=current_grads_log_prob,
          momentum_state_memory=momentum_state_memory,
          step_size=step_size,
          log_accept_ratio=tf.zeros_like(current_target_log_prob,
                                         name='log_accept_ratio'),
          leapfrogs_taken=tf.zeros_like(current_target_log_prob,
                                        dtype=TREE_COUNT_DTYPE,
                                        name='leapfrogs_taken'),
          is_accepted=tf.zeros_like(current_target_log_prob,
                                    dtype=tf.bool,
                                    name='is_accepted'),
          reach_max_depth=tf.zeros_like(current_target_log_prob,
                                        dtype=tf.bool,
                                        name='reach_max_depth'),
          has_divergence=tf.zeros_like(current_target_log_prob,
                                       dtype=tf.bool,
                                       name='has_divergence'),
          energy=compute_hamiltonian(current_target_log_prob, dummy_momentum)
      )


  def loop_tree_doubling(self, step_size, momentum_state_memory,
                         current_step_meta_info, iter_, initial_step_state,
                         initial_step_metastate):
    """Main loop for tree doubling."""
    with tf.name_scope('loop_tree_doubling'):
      batch_shape = prefer_static.shape(current_step_meta_info.init_energy)
      direction = tf.cast(
          tf.random.uniform(
              shape=batch_shape,
              minval=0,
              maxval=2,
              dtype=tf.int32,
              seed=self._seed_stream()),
          dtype=tf.bool)

      tree_start_states = tf.nest.map_structure(
          lambda v: tf.where(  # pylint: disable=g-long-lambda
              _rightmost_expand_to_rank(direction, prefer_static.rank(v[1])),
              v[1], v[0]),
          initial_step_state)

      directions_expanded = [
          _rightmost_expand_to_rank(
              direction, prefer_static.rank(state))
          for state in tree_start_states.state
      ]

      integrator = leapfrog_impl.SimpleLeapfrogIntegrator(
          self.target_log_prob_fn(self.all_states_hack),
          step_sizes=[
              tf.where(d, ss, -ss)
              for d, ss in zip(directions_expanded, step_size)
          ],
          num_steps=self.unrolled_leapfrog_steps)

      [
          candidate_tree_state,
          tree_final_states,
          final_not_divergence,
          continue_tree_final,
          energy_diff_tree_sum,
          momentum_subtree_cumsum,
          leapfrogs_taken
      ] = self._build_sub_tree(
          directions_expanded,
          integrator,
          current_step_meta_info,
          # num_steps_at_this_depth = 2**iter_ = 1 << iter_
          tf.bitwise.left_shift(1, iter_),
          tree_start_states,
          initial_step_metastate.continue_tree,
          initial_step_metastate.not_divergence,
          momentum_state_memory)

      last_candidate_state = initial_step_metastate.candidate_state

      energy_diff_tree_sum = tf.where(
          continue_tree_final,
          energy_diff_tree_sum,
          tf.zeros_like(energy_diff_tree_sum))
      energy_diff_sum = (
          energy_diff_tree_sum + initial_step_metastate.energy_diff_sum)
      if MULTINOMIAL_SAMPLE:
        tree_weight = tf.where(
            continue_tree_final,
            candidate_tree_state.weight,
            tf.constant(-np.inf, dtype=candidate_tree_state.weight.dtype))
        weight_sum = log_add_exp(tree_weight, last_candidate_state.weight)
        log_accept_thresh = tree_weight - last_candidate_state.weight
      else:
        tree_weight = tf.where(
            continue_tree_final,
            candidate_tree_state.weight,
            tf.zeros([], dtype=TREE_COUNT_DTYPE))
        weight_sum = tree_weight + last_candidate_state.weight
        log_accept_thresh = tf.math.log(
            tf.cast(tree_weight, tf.float32) /
            tf.cast(last_candidate_state.weight, tf.float32))
      log_accept_thresh = tf.where(
          tf.math.is_nan(log_accept_thresh),
          tf.zeros([], log_accept_thresh.dtype),
          log_accept_thresh)
      u = tf.math.log1p(-tf.random.uniform(
          shape=batch_shape,
          dtype=log_accept_thresh.dtype,
          seed=self._seed_stream()))
      is_sample_accepted = u <= log_accept_thresh

      choose_new_state = is_sample_accepted & continue_tree_final

      new_candidate_state = TreeDoublingStateCandidate(
          state=[
              tf.where(  # pylint: disable=g-complex-comprehension
                  _rightmost_expand_to_rank(
                      choose_new_state, prefer_static.rank(s0)), s0, s1)
              for s0, s1 in zip(candidate_tree_state.state,
                                last_candidate_state.state)
          ],
          target=tf.where(
              _rightmost_expand_to_rank(
                  choose_new_state,
                  prefer_static.rank(candidate_tree_state.target)),
              candidate_tree_state.target, last_candidate_state.target),
          target_grad_parts=[
              tf.where(  # pylint: disable=g-complex-comprehension
                  _rightmost_expand_to_rank(
                      choose_new_state, prefer_static.rank(grad0)),
                  grad0, grad1)
              for grad0, grad1 in zip(candidate_tree_state.target_grad_parts,
                                      last_candidate_state.target_grad_parts)
          ],
          energy=tf.where(
              _rightmost_expand_to_rank(
                  choose_new_state,
                  prefer_static.rank(candidate_tree_state.target)),
              candidate_tree_state.energy, last_candidate_state.energy),
          weight=weight_sum)

      for new_candidate_state_temp, old_candidate_state_temp in zip(
          new_candidate_state.state, last_candidate_state.state):
        tensorshape_util.set_shape(new_candidate_state_temp,
                                   old_candidate_state_temp.shape)

      for new_candidate_grad_temp, old_candidate_grad_temp in zip(
          new_candidate_state.target_grad_parts,
          last_candidate_state.target_grad_parts):
        tensorshape_util.set_shape(new_candidate_grad_temp,
                                   old_candidate_grad_temp.shape)

      # Update left right information of the trajectory, and check trajectory
      # level U turn
      tree_otherend_states = tf.nest.map_structure(
          lambda v: tf.where(  # pylint: disable=g-long-lambda
              _rightmost_expand_to_rank(direction, prefer_static.rank(v[1])),
              v[0], v[1]), initial_step_state)

      new_step_state = tf.nest.pack_sequence_as(initial_step_state, [
          tf.stack([  # pylint: disable=g-complex-comprehension
              tf.where(
                  _rightmost_expand_to_rank(direction, prefer_static.rank(l)),
                  r, l),
              tf.where(
                  _rightmost_expand_to_rank(direction, prefer_static.rank(l)),
                  l, r),
          ], axis=0)
          for l, r in zip(tf.nest.flatten(tree_final_states),
                          tf.nest.flatten(tree_otherend_states))
      ])

      momentum_tree_cumsum = []
      for p0, p1 in zip(
          initial_step_metastate.momentum_sum, momentum_subtree_cumsum):
        momentum_part_temp = p0 + p1
        tensorshape_util.set_shape(momentum_part_temp, p0.shape)
        momentum_tree_cumsum.append(momentum_part_temp)

      for new_state_temp, old_state_temp in zip(
          tf.nest.flatten(new_step_state),
          tf.nest.flatten(initial_step_state)):
        tensorshape_util.set_shape(new_state_temp, old_state_temp.shape)

      if GENERALIZED_UTURN:
        state_diff = momentum_tree_cumsum
      else:
        state_diff = [s[1] - s[0] for s in new_step_state.state]

      no_u_turns_trajectory = has_not_u_turn(
          state_diff,
          [m[0] for m in new_step_state.momentum],
          [m[1] for m in new_step_state.momentum],
          log_prob_rank=prefer_static.rank_from_shape(batch_shape))

      new_step_metastate = TreeDoublingMetaState(
          candidate_state=new_candidate_state,
          is_accepted=choose_new_state | initial_step_metastate.is_accepted,
          momentum_sum=momentum_tree_cumsum,
          energy_diff_sum=energy_diff_sum,
          continue_tree=continue_tree_final & no_u_turns_trajectory,
          not_divergence=final_not_divergence,
          leapfrog_count=(initial_step_metastate.leapfrog_count +
                          leapfrogs_taken))

      return iter_ + 1, new_step_state, new_step_metastate

