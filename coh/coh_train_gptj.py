import pprint
from functools import partial

from tqdm import tqdm, trange
import numpy as np
import coh.tools.utils as utils

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from flax.training.train_state import TrainState

from coh.data import DatasetOption
from coh.tools.checkpoint import StreamingCheckpointer
from coh.tools.optimizers import OptimizerFactory
from coh.tools.jax_utils import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules,
    cross_entropy_loss_and_accuracy, global_norm, get_float_dtype_by_name,
    set_random_seed, average_metrics, get_weight_decay_mask,
    make_shard_and_gather_fns, with_sharding_constraint,
)
from coh.gptj import GPTJConfig, FlaxGPTJForCausalLMModule


FLAGS, FLAGS_DEF = utils.define_flags_with_default(
    seed=42,
    mesh_dim='1,-1,1',
    dtype='fp32',
    total_steps=10000,
    load_gptj_config='',
    update_gptj_config='',
    load_checkpoint='',
    load_dataset_state='',
    log_freq=50,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    tokenizer=GPTJConfig.get_tokenizer_config(),
    hf_train_dataset=DatasetOption.get_default_config(),
    pt_train_dataset=DatasetOption.get_default_config(),
    hf_eval_dataset=DatasetOption.get_default_config(),
    pt_eval_dataset=DatasetOption.get_default_config(),
    pt_loss_weight=0.01,
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    gptj=GPTJConfig.get_default_config(),
    logger=utils.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),
)


def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    variant = utils.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = utils.user_flags_to_config_dict(FLAGS, FLAGS_DEF)
    logger = utils.WandBLogger(
        config=FLAGS.logger,
        variant=variant,
        enable=FLAGS.log_all_worker or (jax.process_index() == 0),
    )
    set_random_seed(FLAGS.seed)

    if FLAGS.load_dataset_state != '':
        hf, pt = FLAGS.load_dataset_state.split(',')
        hf_dataset = utils.load_pickle(hf)
        pt_dataset = utils.load_pickle(pt)
    else:
        tokenizer = GPTJConfig.get_tokenizer(FLAGS.tokenizer)
        hf_dataset = DatasetOption.load_dataset(FLAGS.hf_train_dataset, tokenizer)
        pt_dataset = DatasetOption.load_dataset(FLAGS.pt_train_dataset, tokenizer)

    if FLAGS.eval_steps > 0:
        hf_eval_dataset = DatasetOption.load_dataset(
            FLAGS.hf_eval_dataset, hf_dataset.tokenizer
        )
        pt_eval_dataset = DatasetOption.load_dataset(
            FLAGS.pt_eval_dataset, pt_dataset.tokenizer
        )
        hf_eval_iterator = iter(hf_eval_dataset)
        pt_eval_iterator = iter(pt_eval_dataset)

    assert hf_dataset.seq_length == pt_dataset.seq_length, "HF and PT datasets must have the same sequence length."
    seq_length = hf_dataset.seq_length

    if FLAGS.load_gptj_config != '':
        gptj_config = GPTJConfig.load_config(FLAGS.load_gptj_config)
    else:
        gptj_config = GPTJConfig(**FLAGS.gptj)

    if FLAGS.update_gptj_config != '':
        gptj_config.update(dict(eval(FLAGS.update_gptj_config)))

    gptj_config.update(dict(
        bos_token_id=hf_dataset.tokenizer.bos_token_id,
        eos_token_id=hf_dataset.tokenizer.eos_token_id,
    ))
    if gptj_config.vocab_size < hf_dataset.vocab_size:
        gptj_config.update(dict(vocab_size=hf_dataset.vocab_size))

    model = FlaxGPTJForCausalLMModule(
        gptj_config, dtype=get_float_dtype_by_name(FLAGS.dtype)
    )

    optimizer, optimizer_info = OptimizerFactory.get_optimizer(
        FLAGS.optimizer,
        get_weight_decay_mask(GPTJConfig.get_weight_decay_exclusions())
    )

    def create_trainstate_from_params(params):
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(gptj_config.rng_keys()),
        )
        return TrainState.create(params=params, tx=optimizer, apply_fn=None)

    def train_step(train_state, rng, hf_batch, pt_batch):
        rng_generator = JaxRNG(rng)
        def loss_and_accuracy(params, batch):
            tokens = with_sharding_constraint(batch['tokens'], PS('dp'))
            loss_masks = with_sharding_constraint(batch['loss_masks'], PS('dp'))
            bos_tokens = jnp.full(
                (tokens.shape[0], 1), gptj_config.bos_token_id, dtype=jnp.int32
            )
            inputs = jnp.concatenate([bos_tokens, tokens[:, :-1]], axis=1)
            logits = model.apply(
                params, inputs, deterministic=False,
                rngs=rng_generator(gptj_config.rng_keys()),
            ).logits
            return cross_entropy_loss_and_accuracy(logits, tokens, loss_masks)
        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True, argnums=0)
        (hf_loss, hf_accuracy), hf_grads = grad_fn(train_state.params, hf_batch)
        (pt_loss, pt_accuracy), pt_grads = grad_fn(train_state.params, pt_batch)
        grads = jax.tree_map(
            lambda hf_grad, pt_grad: hf_grad + pt_grad * FLAGS.pt_loss_weight,
            hf_grads, pt_grads,
        )
        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            hf_loss=hf_loss,
            pt_loss=pt_loss,
            hf_accuracy=hf_accuracy,
            pt_accuracy=pt_accuracy,
            learning_rate=optimizer_info['learning_rate_schedule'](train_state.step),
            gradient_norm=global_norm(grads),
            param_norm=global_norm(train_state.params),
        )
        return train_state, rng_generator(), metrics

    def eval_step(train_state, rng, hf_batch, pt_batch):
        rng_generator = JaxRNG(rng)
        def loss_and_accuracy(params, batch):
            tokens = with_sharding_constraint(batch['tokens'], PS('dp'))
            loss_masks = with_sharding_constraint(batch['loss_masks'], PS('dp'))
            bos_tokens = jnp.full(
                (tokens.shape[0], 1), gptj_config.bos_token_id, dtype=jnp.int32
            )
            inputs = jnp.concatenate([bos_tokens, tokens[:, :-1]], axis=1)
            logits = model.apply(
                params, inputs, deterministic=False,
                rngs=rng_generator(gptj_config.rng_keys()),
            ).logits
            return cross_entropy_loss_and_accuracy(logits, tokens, loss_masks)
        hf_loss, hf_accuracy = loss_and_accuracy(train_state.params, hf_batch)
        pt_loss, pt_accuracy = loss_and_accuracy(train_state.params, pt_batch)
        aux = {
            'hf_accuracy': hf_accuracy,
            'pt_accuracy': pt_accuracy,
            'hf_loss': hf_loss,
            'pt_loss': pt_loss,
        }
        aux = {f'eval_{k}': v for k, v in aux.items()}
        return rng_generator(), metrics

    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state_partition = match_partition_rules(
        GPTJConfig.get_partition_rules(), train_state_shapes
    )

    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )
    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, logger.output_dir,
        enable=jax.process_index() == 0,
    )

    sharded_init_fn = pjit(
        init_fn,
        in_shardings=PS(),
        out_shardings=train_state_partition
    )

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params, ),
        out_shardings=train_state_partition,
        donate_argnums=(0, ),
    )

    sharded_train_step = pjit(
        train_step,
        in_shardings=(train_state_partition, PS(), PS(), PS()),
        out_shardings=(train_state_partition, PS(), PS()),
        donate_argnums=(0, 1),
    )

    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition, PS(), PS(), PS()),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    def save_checkpoint(train_state, milestone=False):
        step = int(jax.device_get(train_state.step))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            gptj_config=gptj_config.to_dict(),
        )
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            milestone=milestone,
        )

    mesh = GPTJConfig.get_jax_mesh(FLAGS.mesh_dim)
    with mesh:
        train_state, restored_params = None, None
        if FLAGS.load_checkpoint != '':
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes, shard_fns
            )

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_fn(next_rng())
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state
            train_state = sharded_create_trainstate_from_params(restored_params)
            del restored_params

        start_step = int(jax.device_get(train_state.step))

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)

        sharded_rng = next_rng()

        step_counter = trange(start_step, FLAGS.total_steps, ncols=0)

        for step, hf_batch, pt_batch in zip(step_counter, hf_dataset, pt_dataset):
            train_state, sharded_rng, metrics = sharded_train_step(
                train_state, sharded_rng, hf_batch, pt_batch
            )

            if step % FLAGS.log_freq == 0:
                if FLAGS.eval_steps > 0:
                    eval_metric_list = []
                    for _ in range(FLAGS.eval_steps):
                        sharded_rng, eval_metrics = sharded_eval_step(
                            train_state, sharded_rng, next(hf_eval_iterator), next(pt_eval_iterator),
                        )
                        eval_metric_list.append(eval_metrics)
                    metrics.update(average_metrics(eval_metric_list))

                log_metrics = {"step": step}
                log_metrics.update(metrics)
                logger.log(log_metrics)
                tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")

            if FLAGS.save_milestone_freq > 0 and (step + 1) % FLAGS.save_milestone_freq == 0:
                save_checkpoint(train_state, milestone=True)
            elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                save_checkpoint(train_state)

        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)


if __name__ == "__main__":
    utils.run(main)
