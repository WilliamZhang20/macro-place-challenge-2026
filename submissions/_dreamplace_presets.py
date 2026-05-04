"""DREAMPlace JSON `extra_params` presets shared by diagnostics and bridge placers."""


def dreamplace_preset_params(preset: str) -> dict:
    if preset == "basic":
        return {}
    if preset == "global_only":
        return {"legalize_flag": 0}
    if preset == "random_global":
        return {"legalize_flag": 0, "random_center_init_flag": 1}
    if preset == "macro":
        return {"macro_place_flag": 1, "two_stage_density_scaler": 1000}
    if preset == "macro_global":
        return {
            "macro_place_flag": 1,
            "two_stage_density_scaler": 1000,
            "legalize_flag": 0,
        }
    if preset == "macro_random_global":
        return {
            "macro_place_flag": 1,
            "two_stage_density_scaler": 1000,
            "legalize_flag": 0,
            "random_center_init_flag": 1,
        }
    if preset == "macro_bb":
        return {
            "macro_place_flag": 1,
            "use_bb": 1,
            "two_stage_density_scaler": 1000,
        }
    if preset == "macro_bb_global":
        return {
            "macro_place_flag": 1,
            "use_bb": 1,
            "two_stage_density_scaler": 1000,
            "legalize_flag": 0,
        }
    if preset == "gift":
        return {"gift_init_flag": 1, "gift_init_scale": 0.7}
    if preset == "routability":
        return {
            "routability_opt_flag": 1,
            "route_num_bins_x": 64,
            "route_num_bins_y": 64,
            "adjust_rudy_area_flag": 1,
            "adjust_pin_area_flag": 1,
        }
    raise ValueError(f"unknown preset {preset!r}")
