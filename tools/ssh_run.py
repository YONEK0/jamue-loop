"""SSH 上机执行工具 (paramiko, 密码认证, 自动试两条线路)。

用法:
  python ssh_run.py "command"                     # 执行单命令, 打印输出
  python ssh_run.py --script local.sh             # 上传并执行本地脚本
  python ssh_run.py --put local remote            # 上传文件
  python ssh_run.py --get remote local            # 下载文件
"""
import argparse
import sys

import paramiko

CREDS = [
    ("223.109.239.32", 13912, "root", "Ahwiugu4"),     # 移动线路 (重启后 v2)
    ("180.127.11.166", 13912, "root", "Ahwiugu4"),     # 电信线路 (重启后 v2)
]


def connect(timeout=15):
    last = None
    for host, port, user, pw in CREDS:
        try:
            cli = paramiko.SSHClient()
            cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            cli.connect(host, port=port, username=user, password=pw,
                        timeout=timeout, banner_timeout=timeout,
                        auth_timeout=timeout, look_for_keys=False,
                        allow_agent=False)
            print(f"[ssh] connected {host}:{port}", file=sys.stderr)
            return cli
        except Exception as e:
            last = e
            print(f"[ssh] {host}:{port} fail: {type(e).__name__}", file=sys.stderr)
    raise SystemExit(f"both routes failed: {last}")


def run(cli, cmd, timeout=120):
    _, out, err = cli.exec_command(cmd, timeout=timeout)
    o = out.read().decode("utf-8", "replace")
    e = err.read().decode("utf-8", "replace")
    rc = out.channel.recv_exit_status()
    return rc, o, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", help="远端命令")
    ap.add_argument("--script", help="上传并执行本地 sh")
    ap.add_argument("--put", nargs=2, metavar=("LOCAL", "REMOTE"))
    ap.add_argument("--get", nargs=2, metavar=("REMOTE", "LOCAL"))
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()
    cli = connect()
    try:
        if args.put:
            sftp = cli.open_sftp()
            sftp.put(args.put[0], args.put[1])
            print(f"[put] {args.put[0]} -> {args.put[1]}")
        elif args.get:
            sftp = cli.open_sftp()
            sftp.get(args.get[0], args.get[1])
            print(f"[get] {args.get[0]} -> {args.get[1]}")
        elif args.script:
            remote = "/tmp/jl_script.sh"
            sftp = cli.open_sftp()
            sftp.put(args.script, remote)
            rc, o, e = run(cli, f"bash {remote}", timeout=args.timeout)
            print(o)
            if e:
                print("[stderr]", e, file=sys.stderr)
            sys.exit(rc)
        else:
            rc, o, e = run(cli, args.cmd, timeout=args.timeout)
            print(o)
            if e:
                print("[stderr]", e, file=sys.stderr)
            sys.exit(rc)
    finally:
        cli.close()


if __name__ == "__main__":
    main()
