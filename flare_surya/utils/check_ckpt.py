import torch

if __name__ == "__main__":

    # ckpt_path = "/anvil/projects/x-cis251356/check_point/surya/baselines/"
    # ckpt_file = "baseline_8hour_zarr_alexnet_lastepoch.ckpt"
    ckpt_path = "/anvil/projects/x-cis251356/check_point/surya/flare/"
    ckpt_file = "last.ckpt"

    ckpt = torch.load(ckpt_path + ckpt_file, map_location="cpu", weights_only=False)

    print(ckpt.keys())
    print(ckpt["epoch"])
