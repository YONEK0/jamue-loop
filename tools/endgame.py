"""收尾执行器: 停训 -> 真实版 eval -> 提示拉回.

用法: python tools/endgame.py
"""
import sys
import time

import paramiko

CREDS = [("223.109.239.30", 31428), ("180.127.11.167", 31428)]


def connect():
    last = None
    for host, port in CREDS:
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(host, port=port, username="root", password="quo3Eenu",
                      timeout=15, look_for_keys=False, allow_agent=False)
            print(f"[ssh] {host}")
            return c
        except Exception as e:
            last = e
    raise SystemExit(f"connect failed: {last}")


def run(c, cmd, t=300):
    _, o, e = c.exec_command(cmd, timeout=t)
    return o.read().decode("utf-8", "replace"), e.read().decode("utf-8", "replace")


def main():
    c = connect()
    step = sys.argv[1] if len(sys.argv) > 1 else "all"

    if step in ("stop", "all"):
        run(c, "pkill -TERM -f '[t]rain_s1_mlm'; sleep 3; pgrep -fc '[t]rain_s1_mlm' || echo stopped")
        print("[endgame] training stopped (rolling ckpt protects)")

    if step in ("eval", "all"):
        # 从 latest 选最新档做 eval
        cmd = ("cd ~/jamue-loop && rm -f results/depth_s1_real2.* && "
               "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup "
               "~/miniconda3/bin/python -u src/eval_depth.py "
               "--ckpt runs/s1_real/ckpt_latest.pt --data data/real_val.bin --data-dtype uint32 "
               "--out results/depth_s1_real2.json --plot results/depth_s1_real2.png "
               "--batches 12 --bs 8 --seq 512 > ~/eval_real2.log 2>&1 < /dev/null & echo eval-bg")
        o, _ = run(c, cmd)
        print("[endgame] eval launched:", o.strip())

    c.close()


if __name__ == "__main__":
    main()
