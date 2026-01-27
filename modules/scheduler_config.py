import copy


from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    DPMSolverSinglestepScheduler,
    EDMDPMSolverMultistepScheduler,
    EDMEulerScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    IPNDMScheduler,
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
        case "ddpm":            return DDPMScheduler
        case "deis":            return DEISMultistepScheduler
        case "dpm_sde":         return DPMSolverSDEScheduler
        case "pndm":            return PNDMScheduler
        case "tcd":             return TCDScheduler
        case "unipc":           return UniPCMultistepScheduler

        case "ipndm":           return IPNDMScheduler

        case _:                 raise NotImplementedError


def get_scheduler_name(scheduler):
    match str(type(scheduler).__name__):
        case "DPMSolverMultistepScheduler":     return "dpmpp_2m"
        # case "DPMSolverMultistepScheduler":     return "dpmpp_2m_sde"
        case "DPMSolverSinglestepScheduler":    return "dpmpp_sde"
        case "KDPM2DiscreteScheduler":          return "dpm_2"
        case "KDPM2AncestralDiscreteScheduler": return "dpm_2_a"
        case "EulerDiscreteScheduler":          return "euler"
        case "EulerAncestralDiscreteScheduler": return "euler_a"
        case "HeunDiscreteScheduler":           return "heun"
        case "LMSDiscreteScheduler":            return "lms"

        case "DDIMScheduler":                   return "ddim"
        case "DDPMScheduler":                   return "ddpm"
        case "DEISMultistepScheduler":          return "deis"
        case "DPMSolverSDEScheduler":           return "dpm_sde"
        case "PNDMScheduler":                   return "pndm"
        case "TCDScheduler":                    return "tcd"
        case "UniPCMultistepScheduler":         return "unipc"

        case "IPNDMScheduler":                  return "ipndm"

        case _:                                 return None


def get_scheduler_supports_setting_timesteps_or_sigmas(scheduler):
    return get_scheduler_supports_setting_timesteps(scheduler) or get_scheduler_supports_setting_sigmas(scheduler)


def get_scheduler_supports_setting_timesteps(scheduler):
    supported = ["dpmpp_2m", "dpmpp_sde", "ddpm", "euler", "heun", "tcd"]
    return get_scheduler_name(scheduler) in supported


def get_scheduler_supports_setting_sigmas(scheduler):
    supported = ["euler"]
    return get_scheduler_name(scheduler) in supported


def get_scheduler_progressbar_offset_index(scheduler, index):
    schedulers = { "heun": 0.5 }
    s = get_scheduler_name(scheduler)
    if s is not None and s in schedulers.keys(): return index * schedulers[s]
    return index


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

