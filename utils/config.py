from model_classes.model import Model
from model_classes.model_class_conditional import ModelClassConditional
from model_classes.model_conditional_batch_norm import ModelConditionalBatchNorm

MODEL_CONFIGS = {
    "Model1": {"class": Model(image_channels=1, nb_channels=64, num_blocks=8, cond_channels=32), "conditional": False,
               "file": "M_ch64_b8_cond32.pth"},
    "Model2": {"class": Model(image_channels=1, nb_channels=128, num_blocks=8, cond_channels=64), "conditional": False,
               "file": "M_ch128_b8_cond64.pth"},
    "ModelConditionalBatchNorm1": {"class": ModelConditionalBatchNorm(image_channels=1, nb_channels=128, num_blocks=6, cond_channels=32),
                                   "conditional": False, "file": "MCBN_ch128_b6_cond32.pth"},
    "ModelConditionalBatchNorm2": {"class": ModelConditionalBatchNorm(image_channels=1, nb_channels=128, num_blocks=8, cond_channels=64),
                                   "conditional": False, "file": "MCBN_ch128_b8_cond64.pth"},
    "ModelClassConditional1": {"class": ModelClassConditional(image_channels=1, nb_channels=128, num_blocks=6, cond_channels=32, num_classes=10),
                               "conditional": True, "file": "MCC_ch128_b6_cond32.pth"},
    "ModelClassConditional2": {"class": ModelClassConditional(image_channels=1, nb_channels=128, num_blocks=8, cond_channels=64, num_classes=10),
                               "conditional": True, "file": "MCC_ch128_b8_cond64.pth"},
    "ModelClassConditional3": {"class": ModelClassConditional(image_channels=1, nb_channels=128, num_blocks=6, cond_channels=32, num_classes=10),
                               "conditional": True, "file": "MCC_ch128_b6_cond32_e100.pth"},
    "ModelClassConditional4": {"class": ModelClassConditional(image_channels=1, nb_channels=128, num_blocks=8, cond_channels=64, num_classes=10),
                               "conditional": True, "file": "MCC_ch128_b8_cond64_e100.pth"},
}
