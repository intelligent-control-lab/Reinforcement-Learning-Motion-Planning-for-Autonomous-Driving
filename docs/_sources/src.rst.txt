:mod:`src`
==================================================

.. automodule:: src
.. toctree::

    src/configs
    src/models
    src/README
    src/utils
    src/ddpg
    src/train
    src/external
    src/modules

.. automodule:: src.configs
.. rubric:: :doc:`src/configs`

.. autosummary::
    :nosignatures:

    add_experiment_args
    add_logging_args
    add_training_args
    add_encoder_args
    add_rl_agent_args
    add_car_env_args
    add_spinning_up_args

.. automodule:: src.models
.. rubric:: :doc:`src/models`

.. autosummary::
    :nosignatures:

    fanin_init
    Actor
    Critic


.. automodule:: src.utils
.. rubric:: :doc:`src/utils`

.. autosummary::
    :nosignatures:

    to_numpy
    to_tensor
    soft_update
    hard_update
    load_experiment_settings
    colorize
    log
    path_exists
    mkdir_p

.. automodule:: src.ddpg
.. rubric:: :doc:`src/ddpg`

.. autosummary::
    :nosignatures:

    DDPG

.. automodule:: src.train
.. rubric:: :doc:`src/train`

.. autosummary::
    :nosignatures:

    train
    experiment

