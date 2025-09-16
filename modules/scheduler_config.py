import copy


from diffusers.schedulers import (
    DDIMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    PNDMScheduler,
    TCDScheduler,
    UniPCMultistepScheduler,
)


def get_scheduler(scheduler_dict, current_scheduler_config):
    scheduler_config = get_scheduler_config(scheduler_dict, current_scheduler_config)
    scheduler_class = get_scheduler_class(scheduler_dict["scheduler"])
    return scheduler_class.from_config(scheduler_config)


def get_scheduler_class(scheduler_name):
    match scheduler_name:
        case "dpmpp_2m":        return DPMSolverMultistepScheduler
        case "dpmpp_2m_sde":    return DPMSolverMultistepScheduler
        case "dpmpp_sde":       return DPMSolverSinglestepScheduler
        case "dpm_2":           return KDPM2DiscreteScheduler
        case "dpm_2_a":         return KDPM2AncestralDiscreteScheduler
        case "euler":           return EulerDiscreteScheduler
        case "euler_a":         return EulerAncestralDiscreteScheduler
        case "heun":            return HeunDiscreteScheduler
        case "lms":             return LMSDiscreteScheduler

        case "ddim":            return DDIMScheduler
        case "deis":            return DEISMultistepScheduler
        case "dpm_sde":         return DPMSolverSDEScheduler
        case "pndm":            return PNDMScheduler
        case "tcd":             return TCDScheduler
        case "unipc":           return UniPCMultistepScheduler

        case _:                 raise NotImplementedError


def get_scheduler_config(scheduler_dict, current_scheduler_config):
    for k, v in scheduler_dict.items():
        if k == "scheduler": continue
        current_scheduler_config[k] = v
    match scheduler_dict["scheduler"]:
        case "dpmpp_2m":
            current_scheduler_config["algorithm_type"] = "dpmsolver++"
            current_scheduler_config["solver_order"] = 2
        case "dpmpp_2m_sde":
            current_scheduler_config["algorithm_type"] = "sde-dpmsolver++"
    return current_scheduler_config

