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
    FlowMatchEulerDiscreteScheduler,
    FlowMatchHeunDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    PNDMScheduler,
    TCDScheduler,
    UniPCMultistepScheduler,
)


def get_scheduler_class(scheduler_name):
    match scheduler_name:
        case "DDIM":            return DDIMScheduler
        case "DDPM":            return DDPMScheduler
        case "DEIS":            return DEISMultistepScheduler
        case "DPM++ 2M":        return DPMSolverMultistepScheduler
        case "DPM++ SDE":       return DPMSolverSinglestepScheduler
        case "DPM SDE":         return DPMSolverSDEScheduler
        case "DPM2":            return KDPM2DiscreteScheduler
        case "DPM2 a":          return KDPM2AncestralDiscreteScheduler
        case "Euler":           return EulerDiscreteScheduler
        case "Euler a":         return EulerAncestralDiscreteScheduler
        case "FM Euler":        return FlowMatchEulerDiscreteScheduler
        case "FM Heun":         return FlowMatchHeunDiscreteScheduler
        case "Heun":            return HeunDiscreteScheduler
        case "LMS":             return LMSDiscreteScheduler
        case "PNDM":            return PNDMScheduler
        case "TCD":             return TCDScheduler
        case "UniPC":           return UniPCMultistepScheduler
        case _:                 raise NotImplementedError


def get_scheduler_name(scheduler):
    match str(type(scheduler).__name__):
        case "DDIMScheduler":                   return "DDIM"
        case "DDPMScheduler":                   return "DDPM"
        case "DEISMultistepScheduler":          return "DEIS"
        case "DPMSolverMultistepScheduler":     return "DPM++ 2M"
        case "DPMSolverSinglestepScheduler":    return "DPM++ SDE"
        case "DPMSolverSDEScheduler":           return "DPM SDE"
        case "KDPM2DiscreteScheduler":          return "DPM2"
        case "KDPM2AncestralDiscreteScheduler": return "DPM2 a"
        case "EulerDiscreteScheduler":          return "Euler"
        case "EulerAncestralDiscreteScheduler": return "Euler a"
        case "FlowMatchEulerDiscreteScheduler": return "FM Euler"
        case "FlowMatchHeunDiscreteScheduler":  return "FM Heun"
        case "HeunDiscreteScheduler":           return "Heun"
        case "LMSDiscreteScheduler":            return "LMS"
        case "PNDMScheduler":                   return "PNDM"
        case "TCDScheduler":                    return "TCD"
        case "UniPCMultistepScheduler":         return "UniPC"
        case _:                                 raise NotImplementedError


def get_scheduler_supports_setting_timesteps_or_sigmas(scheduler):
    return get_scheduler_supports_setting_timesteps(scheduler) or get_scheduler_supports_setting_sigmas(scheduler)


def get_scheduler_supports_setting_timesteps(scheduler):
    supported = ["DDPM", "DPM++ 2M", "DPM++ SDE", "Euler", "Heun", "TCD"]
    return get_scheduler_name(scheduler) in supported


def get_scheduler_supports_setting_sigmas(scheduler):
    supported = ["Euler"]
    return get_scheduler_name(scheduler) in supported
