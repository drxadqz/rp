from __future__ import annotations

import sys
import torch


def main() -> int:
    print("python:", sys.executable)
    print("torch:", torch.__version__)
    print("torch cuda build:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        return 1
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("cuda matmul ok:", float(y.mean().detach().cpu()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

