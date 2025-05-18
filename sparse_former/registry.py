import mmdet.registry as mmdet_registry
from mmdet.registry import Registry

# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry('runner', parent=mmdet_registry.RUNNERS)
# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry(
    'runner constructor', parent=mmdet_registry.RUNNER_CONSTRUCTORS
)
# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry('loop', parent=mmdet_registry.LOOPS)
# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry(
    'hook', 
    parent=mmdet_registry.HOOKS,
)

# # manage data-related modules
DATASETS = Registry(
    'dataset', 
    parent=mmdet_registry.DATASETS,
)
DATA_SAMPLERS = Registry('data sampler', parent=mmdet_registry.DATA_SAMPLERS)
TRANSFORMS = Registry(
    'transform', 
    parent=mmdet_registry.TRANSFORMS, 
)

# manage all kinds of modules inheriting `nn.Module`
MODELS = Registry(
    'model', 
    parent=mmdet_registry.MODELS, 
    locations=['sparse_former.models']
)
# manage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry('model_wrapper', parent=mmdet_registry.MODEL_WRAPPERS)
# manage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry(
    'weight initializer', parent=mmdet_registry.WEIGHT_INITIALIZERS
)

# manage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry('optimizer', parent=mmdet_registry.OPTIMIZERS)
# manage optimizer wrapper
OPTIM_WRAPPERS = Registry('optim_wrapper', parent=mmdet_registry.OPTIM_WRAPPERS)
# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer constructor', parent=mmdet_registry.OPTIM_WRAPPER_CONSTRUCTORS
)
# manage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = Registry(
    'parameter scheduler', parent=mmdet_registry.PARAM_SCHEDULERS
)
# manage all kinds of metrics
METRICS = Registry(
    'metric', 
    parent=mmdet_registry.METRICS,
    locations=['sparse_former.evaluation.metrics'] 
)

# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry(
    'task util', 
    parent=mmdet_registry.TASK_UTILS,
    locations=['sparse_former.models.task_modules'] 
)

# manage visualizer
VISUALIZERS = Registry(
    'visualizer',
    parent=mmdet_registry.VISUALIZERS,
)
# manage visualizer backend
VISBACKENDS = Registry('vis_backend', parent=mmdet_registry.VISBACKENDS)

# manage logprocessor
LOG_PROCESSORS = Registry('log_processor', parent=mmdet_registry.LOG_PROCESSORS)
