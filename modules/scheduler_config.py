from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)


def get_scheduler(scheduler_name, current_scheduler_config):
    scheduler_config = get_scheduler_config(scheduler_name, current_scheduler_config)
    scheduler_class = get_scheduler_class(scheduler_name)
    return scheduler_class.from_config(scheduler_config)


def get_scheduler_class(scheduler_name):
    name = scheduler_name.replace("k_", "", 1)
    match name:
        case "ddim":            return DDIMScheduler
        case "euler":           return EulerDiscreteScheduler
        case "euler_a":         return EulerAncestralDiscreteScheduler
        case "dpm_2":           return KDPM2DiscreteScheduler
        case "dpm_2_a":         return KDPM2AncestralDiscreteScheduler
        case "dpmpp_2m":        return DPMSolverMultistepScheduler
        case "dpmpp_2m_sde":    return DPMSolverMultistepScheduler
        case "dpmpp_sde":       return DPMSolverSinglestepScheduler
        case "heun":            return HeunDiscreteScheduler
        case "lms":             return LMSDiscreteScheduler
        case "pndm":            return PNDMScheduler
        case "unipc":           return UniPCMultistepScheduler
        case _:                 raise NotImplementedError


def get_scheduler_config(scheduler_name, current_scheduler_config):
    name = scheduler_name
    if name.startswith("k_"):
        current_scheduler_config["use_karras_sigmas"] = True
        name = scheduler_name.replace("k_", "", 1)
    match name:
        case "dpmpp_2m":
            current_scheduler_config["algorithm_type"] = "dpmsolver++"
            current_scheduler_config["solver_order"] = 2
        case "dpmpp_2m_sde":
            current_scheduler_config["algorithm_type"] = "sde-dpmsolver++"
    return current_scheduler_config

